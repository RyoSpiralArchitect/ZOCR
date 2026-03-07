from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from .metadata import build_job_repository
from .policy import build_audit_logger, build_tenant_approval_notifier
from .storage import build_job_store


def _build_repository():
    _storage_backend, job_store = build_job_store()
    _metadata_backend, job_repository = build_job_repository(job_store)
    return job_repository


def _build_audit_logger():
    return build_audit_logger(default_dsn=os.environ.get("ZOCR_API_DATABASE_URL") or "")[1]


def _json_dump(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _cli_subject() -> str:
    return os.environ.get("ZOCR_ADMIN_SUBJECT") or getpass.getuser() or "zocr-admin"


def _audit_cli(logger, event: str, payload: dict[str, Any]) -> None:
    if not logger.enabled:
        return
    logger.write(
        event,
        {
            "auth_mode": "cli",
            "subject": _cli_subject(),
            "source": "cli",
            "is_admin": True,
            **payload,
        },
    )


def _optional_int(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(0, int(value))


def _parse_datetime_arg(name: str, value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception as exc:
        raise SystemExit(f"Invalid {name} timestamp") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _plan_payload_from_args(args) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.max_active_jobs is not None:
        payload["max_active_jobs"] = _optional_int(args.max_active_jobs)
    if args.max_stored_jobs is not None:
        payload["max_stored_jobs"] = _optional_int(args.max_stored_jobs)
    if args.rate_limit_per_min is not None:
        payload["rate_limit_per_min"] = _optional_int(args.rate_limit_per_min)
    return payload


def _policy_payload_from_args(args) -> dict[str, Any]:
    payload = _plan_payload_from_args(args)
    if args.plan is not None:
        payload["plan_name"] = str(args.plan).strip() or None
    return payload


def _change_request_response(change_request) -> dict[str, Any]:
    return {"change_request": change_request.__dict__}


def _change_request_approval_event(change_request) -> str:
    if str(getattr(change_request, "status", "")) == "approved":
        return "tenant_change_request.approved"
    return "tenant_change_request.reviewed"


def _notify_change_request(notifier, event: str, change_request) -> None:
    if not notifier.enabled:
        return
    notifier.notify(
        event,
        {
            "source": "cli",
            "tenant_id": change_request.target_id if change_request.target_type == "policy" else None,
            "actor": {
                "subject": _cli_subject(),
                "auth_mode": "cli",
                "is_admin": True,
            },
            "change_request": change_request.__dict__,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("zocr-admin")
    subparsers = parser.add_subparsers(dest="resource", required=True)

    plans_parser = subparsers.add_parser("tenant-plan")
    plans_subparsers = plans_parser.add_subparsers(dest="action", required=True)
    plans_subparsers.add_parser("list")
    plan_put = plans_subparsers.add_parser("put")
    plan_put.add_argument("plan_name")
    plan_put.add_argument("--max-active-jobs", type=int)
    plan_put.add_argument("--max-stored-jobs", type=int)
    plan_put.add_argument("--rate-limit-per-min", type=int)
    plan_delete = plans_subparsers.add_parser("delete")
    plan_delete.add_argument("plan_name")

    policies_parser = subparsers.add_parser("tenant-policy")
    policies_subparsers = policies_parser.add_subparsers(dest="action", required=True)
    policies_subparsers.add_parser("list")
    policy_put = policies_subparsers.add_parser("put")
    policy_put.add_argument("tenant_id")
    policy_put.add_argument("--plan")
    policy_put.add_argument("--max-active-jobs", type=int)
    policy_put.add_argument("--max-stored-jobs", type=int)
    policy_put.add_argument("--rate-limit-per-min", type=int)
    policy_delete = policies_subparsers.add_parser("delete")
    policy_delete.add_argument("tenant_id")

    audit_parser = subparsers.add_parser("audit")
    audit_subparsers = audit_parser.add_subparsers(dest="action", required=True)
    audit_list = audit_subparsers.add_parser("list")
    audit_list.add_argument("--tenant-id")
    audit_list.add_argument("--event")
    audit_list.add_argument("--subject")
    audit_list.add_argument("--request-id")
    audit_list.add_argument("--contains")
    audit_list.add_argument("--since")
    audit_list.add_argument("--until")
    audit_list.add_argument("--limit", type=int, default=100)

    requests_parser = subparsers.add_parser("tenant-request")
    requests_subparsers = requests_parser.add_subparsers(dest="action", required=True)
    request_list = requests_subparsers.add_parser("list")
    request_list.add_argument("--status")
    request_list.add_argument("--target-type")
    request_list.add_argument("--limit", type=int, default=100)
    request_approve = requests_subparsers.add_parser("approve")
    request_approve.add_argument("request_id")
    request_approve.add_argument("--reason")
    request_reject = requests_subparsers.add_parser("reject")
    request_reject.add_argument("request_id")
    request_reject.add_argument("--reason")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo = _build_repository()
    audit_logger = _build_audit_logger()
    tenant_approval_notifier = build_tenant_approval_notifier()[1]
    approval_required = _env_truthy("ZOCR_API_TENANT_APPROVAL_REQUIRED", False)

    if args.resource == "tenant-plan" and args.action == "list":
        plans = [plan.__dict__ for plan in repo.list_tenant_plans()]
        _audit_cli(audit_logger, "tenant_plan.listed", {"count": len(plans)})
        _json_dump({"plans": plans, "count": len(plans)})
        return
    if args.resource == "tenant-plan" and args.action == "put":
        if approval_required:
            change_request = repo.create_tenant_change_request(
                {
                    "target_type": "plan",
                    "target_id": args.plan_name,
                    "action": "upsert",
                    "payload": _plan_payload_from_args(args),
                    "requested_by": _cli_subject(),
                }
            )
            _audit_cli(
                audit_logger,
                "tenant_change_request.created",
                {
                    "change_request_id": change_request.request_id,
                    "target_type": change_request.target_type,
                    "target_id": change_request.target_id,
                    "action": change_request.action,
                    "status": change_request.status,
                },
            )
            _notify_change_request(tenant_approval_notifier, "tenant_change_request.created", change_request)
            _json_dump(_change_request_response(change_request))
            return
        plan = repo.upsert_tenant_plan(args.plan_name, _plan_payload_from_args(args))
        _audit_cli(audit_logger, "tenant_plan.upserted", {"plan_name": plan.plan_name, "plan": plan.__dict__})
        _json_dump({"plan": plan.__dict__})
        return
    if args.resource == "tenant-plan" and args.action == "delete":
        if approval_required:
            existing = next((plan for plan in repo.list_tenant_plans() if plan.plan_name == args.plan_name), None)
            if existing is None or existing.source == "env":
                raise SystemExit("Unknown tenant plan")
            change_request = repo.create_tenant_change_request(
                {
                    "target_type": "plan",
                    "target_id": args.plan_name,
                    "action": "delete",
                    "requested_by": _cli_subject(),
                }
            )
            _audit_cli(
                audit_logger,
                "tenant_change_request.created",
                {
                    "change_request_id": change_request.request_id,
                    "target_type": change_request.target_type,
                    "target_id": change_request.target_id,
                    "action": change_request.action,
                    "status": change_request.status,
                },
            )
            _notify_change_request(tenant_approval_notifier, "tenant_change_request.created", change_request)
            _json_dump(_change_request_response(change_request))
            return
        deleted = repo.delete_tenant_plan(args.plan_name)
        if not deleted:
            raise SystemExit("Unknown tenant plan")
        _audit_cli(audit_logger, "tenant_plan.deleted", {"plan_name": args.plan_name})
        _json_dump({"ok": True, "plan_name": args.plan_name})
        return
    if args.resource == "tenant-policy" and args.action == "list":
        policies = [policy.__dict__ for policy in repo.list_tenant_policies()]
        _audit_cli(audit_logger, "tenant_policy.listed", {"count": len(policies)})
        _json_dump({"policies": policies, "count": len(policies)})
        return
    if args.resource == "tenant-policy" and args.action == "put":
        if approval_required:
            change_request = repo.create_tenant_change_request(
                {
                    "target_type": "policy",
                    "target_id": args.tenant_id,
                    "action": "upsert",
                    "payload": _policy_payload_from_args(args),
                    "requested_by": _cli_subject(),
                }
            )
            _audit_cli(
                audit_logger,
                "tenant_change_request.created",
                {
                    "change_request_id": change_request.request_id,
                    "target_type": change_request.target_type,
                    "target_id": change_request.target_id,
                    "action": change_request.action,
                    "status": change_request.status,
                },
            )
            _notify_change_request(tenant_approval_notifier, "tenant_change_request.created", change_request)
            _json_dump(_change_request_response(change_request))
            return
        policy = repo.upsert_tenant_policy(args.tenant_id, _policy_payload_from_args(args))
        _audit_cli(
            audit_logger,
            "tenant_policy.upserted",
            {"tenant_id": policy.tenant_id, "policy": policy.__dict__},
        )
        _json_dump({"policy": policy.__dict__})
        return
    if args.resource == "tenant-policy" and args.action == "delete":
        if approval_required:
            existing = repo.resolve_tenant_policy(args.tenant_id)
            if existing is None or existing.source == "env":
                raise SystemExit("Unknown tenant policy")
            change_request = repo.create_tenant_change_request(
                {
                    "target_type": "policy",
                    "target_id": args.tenant_id,
                    "action": "delete",
                    "requested_by": _cli_subject(),
                }
            )
            _audit_cli(
                audit_logger,
                "tenant_change_request.created",
                {
                    "change_request_id": change_request.request_id,
                    "target_type": change_request.target_type,
                    "target_id": change_request.target_id,
                    "action": change_request.action,
                    "status": change_request.status,
                },
            )
            _notify_change_request(tenant_approval_notifier, "tenant_change_request.created", change_request)
            _json_dump(_change_request_response(change_request))
            return
        deleted = repo.delete_tenant_policy(args.tenant_id)
        if not deleted:
            raise SystemExit("Unknown tenant policy")
        _audit_cli(audit_logger, "tenant_policy.deleted", {"tenant_id": args.tenant_id})
        _json_dump({"ok": True, "tenant_id": args.tenant_id})
        return
    if args.resource == "tenant-request" and args.action == "list":
        try:
            requests = [
                item.__dict__
                for item in repo.list_tenant_change_requests(
                    status=str(args.status).strip() or None if args.status is not None else None,
                    target_type=str(args.target_type).strip() or None if args.target_type is not None else None,
                    limit=max(1, min(int(args.limit), 200)),
                )
            ]
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        _audit_cli(
            audit_logger,
            "tenant_change_request.listed",
            {
                "count": len(requests),
                "filters": {
                    "status": str(args.status).strip() or None if args.status is not None else None,
                    "target_type": str(args.target_type).strip() or None if args.target_type is not None else None,
                    "limit": max(1, min(int(args.limit), 200)),
                },
            },
        )
        _json_dump({"change_requests": requests, "count": len(requests)})
        return
    if args.resource == "tenant-request" and args.action == "approve":
        try:
            change_request = repo.approve_tenant_change_request(
                args.request_id,
                reviewed_by=_cli_subject(),
                review_reason=str(args.reason).strip() or None if args.reason is not None else None,
            )
        except (PermissionError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if change_request is None:
            raise SystemExit("Unknown tenant change request")
        event = _change_request_approval_event(change_request)
        _audit_cli(
            audit_logger,
            event,
            {
                "change_request_id": change_request.request_id,
                "target_type": change_request.target_type,
                "target_id": change_request.target_id,
                "action": change_request.action,
                "status": change_request.status,
                "approvals_required": getattr(change_request, "approvals_required", 1),
                "approvals_received": getattr(change_request, "approvals_received", 0),
                "review_reason": change_request.review_reason,
            },
        )
        _notify_change_request(tenant_approval_notifier, event, change_request)
        _json_dump(_change_request_response(change_request))
        return
    if args.resource == "tenant-request" and args.action == "reject":
        try:
            change_request = repo.reject_tenant_change_request(
                args.request_id,
                reviewed_by=_cli_subject(),
                review_reason=str(args.reason).strip() or None if args.reason is not None else None,
            )
        except (PermissionError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        if change_request is None:
            raise SystemExit("Unknown tenant change request")
        _audit_cli(
            audit_logger,
            "tenant_change_request.rejected",
            {
                "change_request_id": change_request.request_id,
                "target_type": change_request.target_type,
                "target_id": change_request.target_id,
                "action": change_request.action,
                "status": change_request.status,
                "review_reason": change_request.review_reason,
            },
        )
        _notify_change_request(tenant_approval_notifier, "tenant_change_request.rejected", change_request)
        _json_dump(_change_request_response(change_request))
        return
    if args.resource == "audit" and args.action == "list":
        if not audit_logger.readable:
            raise SystemExit("Audit read backend is not configured")
        since_dt = _parse_datetime_arg("since", args.since)
        until_dt = _parse_datetime_arg("until", args.until)
        if since_dt and until_dt and since_dt > until_dt:
            raise SystemExit("`since` must be earlier than `until`")
        events = audit_logger.search(
            limit=max(1, min(int(args.limit), 200)),
            tenant_id=str(args.tenant_id).strip() or None if args.tenant_id is not None else None,
            event=str(args.event).strip() or None if args.event is not None else None,
            subject=str(args.subject).strip() or None if args.subject is not None else None,
            request_id=str(args.request_id).strip() or None if args.request_id is not None else None,
            contains=str(args.contains).strip() or None if args.contains is not None else None,
            since=since_dt,
            until=until_dt,
        )
        _audit_cli(
            audit_logger,
            "audit_event.listed",
            {"count": len(events), "backend": audit_logger.read_backend, "source": "cli"},
        )
        _json_dump({"events": events, "count": len(events), "backend": audit_logger.read_backend})
        return

    raise SystemExit("Unsupported command")
