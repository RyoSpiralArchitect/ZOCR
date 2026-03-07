from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Protocol

from .storage import JobStore

_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class JobRecordRef:
    job_id: str
    tenant_id: str
    last_modified: float


@dataclass(frozen=True)
class TenantPolicy:
    tenant_id: str
    plan_name: Optional[str] = None
    max_active_jobs: Optional[int] = None
    max_stored_jobs: Optional[int] = None
    rate_limit_per_min: Optional[int] = None
    source: str = "env"


@dataclass(frozen=True)
class TenantPlan:
    plan_name: str
    max_active_jobs: Optional[int] = None
    max_stored_jobs: Optional[int] = None
    rate_limit_per_min: Optional[int] = None
    source: str = "env"


@dataclass(frozen=True)
class TenantChangeRequest:
    request_id: str
    target_type: str
    target_id: str
    action: str
    payload: Dict[str, Any]
    status: str
    approvals_required: int = 1
    approvals_received: int = 0
    approval_records: tuple[Dict[str, Any], ...] = ()
    requested_by: Optional[str] = None
    requested_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_reason: Optional[str] = None
    source: str = "local"


class JobRepository(Protocol):
    backend: str

    def list_job_refs(
        self,
        tenant_id: Optional[str] = None,
        *,
        limit: Optional[int] = None,
    ) -> list[JobRecordRef]: ...

    def count_jobs(self, tenant_id: str, *, statuses: Optional[tuple[str, ...]] = None) -> int: ...
    def list_tenant_plans(self) -> list[TenantPlan]: ...
    def upsert_tenant_plan(self, plan_name: str, payload: Dict[str, Any]) -> TenantPlan: ...
    def delete_tenant_plan(self, plan_name: str) -> bool: ...
    def list_tenant_policies(self) -> list[TenantPolicy]: ...
    def upsert_tenant_policy(self, tenant_id: str, payload: Dict[str, Any]) -> TenantPolicy: ...
    def delete_tenant_policy(self, tenant_id: str) -> bool: ...
    def list_tenant_change_requests(
        self,
        *,
        status: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TenantChangeRequest]: ...
    def read_tenant_change_request(self, request_id: str) -> Optional[TenantChangeRequest]: ...
    def create_tenant_change_request(self, payload: Dict[str, Any]) -> TenantChangeRequest: ...
    def approve_tenant_change_request(
        self,
        request_id: str,
        *,
        reviewed_by: str,
        review_reason: Optional[str] = None,
    ) -> Optional[TenantChangeRequest]: ...
    def reject_tenant_change_request(
        self,
        request_id: str,
        *,
        reviewed_by: str,
        review_reason: Optional[str] = None,
    ) -> Optional[TenantChangeRequest]: ...
    def resolve_tenant_policy(self, tenant_id: str) -> Optional[TenantPolicy]: ...
    def read_job(self, job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]: ...
    def write_job(self, job_id: str, tenant_id: str, payload: Dict[str, Any]) -> None: ...
    def delete_job(self, job_id: str, tenant_id: str) -> None: ...


def normalize_tenant_id(value: str) -> str:
    tenant_id = str(value or "").strip()
    if not tenant_id:
        raise ValueError("Tenant id must not be empty.")
    if not _TENANT_ID_RE.match(tenant_id):
        raise ValueError(
            "Tenant id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}."
        )
    return tenant_id


def default_tenant_from_env(*, multi_tenant_enabled: bool = False) -> str:
    raw = os.environ.get("ZOCR_API_DEFAULT_TENANT")
    if raw is None:
        return "" if multi_tenant_enabled else "default"
    try:
        return normalize_tenant_id(raw)
    except ValueError:
        return ""


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _coerce_optional_limit(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except Exception:
        return None


def _normalize_plan_name(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    return raw or None


def _normalize_tenant_plan_payload(plan_name: str, payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Tenant plan payload must be an object.")
    normalized_name = _normalize_plan_name(plan_name)
    if not normalized_name:
        raise RuntimeError("Tenant plan name must not be empty.")
    return {
        "plan_name": normalized_name,
        "max_active_jobs": _coerce_optional_limit(payload.get("max_active_jobs")),
        "max_stored_jobs": _coerce_optional_limit(payload.get("max_stored_jobs")),
        "rate_limit_per_min": _coerce_optional_limit(payload.get("rate_limit_per_min")),
    }


def _normalize_tenant_policy_payload(tenant_id: str, payload: Any) -> Dict[str, Any]:
    normalized_tenant = normalize_tenant_id(tenant_id)
    if isinstance(payload, str):
        return {"tenant_id": normalized_tenant, "plan_name": _normalize_plan_name(payload)}
    if not isinstance(payload, dict):
        raise RuntimeError("Tenant policy payload must be an object or plan string.")
    return {
        "tenant_id": normalized_tenant,
        "plan_name": _normalize_plan_name(payload.get("plan") or payload.get("plan_name")),
        "max_active_jobs": _coerce_optional_limit(payload.get("max_active_jobs")),
        "max_stored_jobs": _coerce_optional_limit(payload.get("max_stored_jobs")),
        "rate_limit_per_min": _coerce_optional_limit(payload.get("rate_limit_per_min")),
    }


def _load_tenant_plans_from_env() -> dict[str, Dict[str, Any]]:
    raw = (os.environ.get("ZOCR_API_TENANT_PLANS_JSON") or "").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ZOCR_API_TENANT_PLANS_JSON must be valid JSON.") from exc
    plans: dict[str, Dict[str, Any]] = {}
    if isinstance(loaded, dict):
        for plan_name, payload in loaded.items():
            plan = _normalize_tenant_plan_payload(str(plan_name), payload)
            plans[str(plan["plan_name"])] = plan
        return plans
    if isinstance(loaded, list):
        for item in loaded:
            if not isinstance(item, dict):
                raise RuntimeError("ZOCR_API_TENANT_PLANS_JSON list entries must be objects.")
            plan_name = _normalize_plan_name(item.get("name") or item.get("plan_name"))
            if not plan_name:
                raise RuntimeError("Tenant plan entries require `name` or `plan_name`.")
            plan = _normalize_tenant_plan_payload(plan_name, item)
            plans[str(plan["plan_name"])] = plan
        return plans
    raise RuntimeError("ZOCR_API_TENANT_PLANS_JSON must be a JSON object or array.")


def _load_tenant_policies_from_env() -> dict[str, Dict[str, Any]]:
    raw = (os.environ.get("ZOCR_API_TENANT_POLICIES_JSON") or "").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ZOCR_API_TENANT_POLICIES_JSON must be valid JSON.") from exc
    policies: dict[str, Dict[str, Any]] = {}
    if isinstance(loaded, dict):
        for tenant_id, payload in loaded.items():
            policy = _normalize_tenant_policy_payload(str(tenant_id), payload)
            policies[str(policy["tenant_id"])] = policy
        return policies
    if isinstance(loaded, list):
        for item in loaded:
            if not isinstance(item, dict):
                raise RuntimeError("ZOCR_API_TENANT_POLICIES_JSON list entries must be objects.")
            tenant_id = item.get("tenant_id")
            if not tenant_id:
                raise RuntimeError("Tenant policy entries require `tenant_id`.")
            policy = _normalize_tenant_policy_payload(str(tenant_id), item)
            policies[str(policy["tenant_id"])] = policy
        return policies
    raise RuntimeError("ZOCR_API_TENANT_POLICIES_JSON must be a JSON object or array.")


def _resolve_tenant_policy_record(
    *,
    tenant_id: str,
    source: str,
    plan_payload: Optional[Dict[str, Any]],
    tenant_payload: Optional[Dict[str, Any]],
) -> Optional[TenantPolicy]:
    normalized_tenant = normalize_tenant_id(tenant_id)
    plan_name = _normalize_plan_name((tenant_payload or {}).get("plan_name")) or _normalize_plan_name(
        (plan_payload or {}).get("plan_name")
    )
    max_active_jobs = (tenant_payload or {}).get("max_active_jobs")
    if max_active_jobs is None:
        max_active_jobs = (plan_payload or {}).get("max_active_jobs")
    max_stored_jobs = (tenant_payload or {}).get("max_stored_jobs")
    if max_stored_jobs is None:
        max_stored_jobs = (plan_payload or {}).get("max_stored_jobs")
    rate_limit_per_min = (tenant_payload or {}).get("rate_limit_per_min")
    if rate_limit_per_min is None:
        rate_limit_per_min = (plan_payload or {}).get("rate_limit_per_min")
    if plan_name is None and max_active_jobs is None and max_stored_jobs is None and rate_limit_per_min is None:
        return None
    return TenantPolicy(
        tenant_id=normalized_tenant,
        plan_name=plan_name,
        max_active_jobs=_coerce_optional_limit(max_active_jobs),
        max_stored_jobs=_coerce_optional_limit(max_stored_jobs),
        rate_limit_per_min=_coerce_optional_limit(rate_limit_per_min),
        source=source,
    )


def _tenant_plan_from_payload(plan_name: str, payload: Dict[str, Any], *, source: str) -> TenantPlan:
    normalized_name = _normalize_plan_name(plan_name) or ""
    return TenantPlan(
        plan_name=normalized_name,
        max_active_jobs=_coerce_optional_limit(payload.get("max_active_jobs")),
        max_stored_jobs=_coerce_optional_limit(payload.get("max_stored_jobs")),
        rate_limit_per_min=_coerce_optional_limit(payload.get("rate_limit_per_min")),
        source=source,
    )


def _normalize_change_request_status(value: Any) -> str:
    status = str(value or "pending").strip().lower()
    if status not in {"pending", "approved", "rejected"}:
        raise ValueError("Change request status must be one of pending, approved, rejected.")
    return status


def _normalize_change_request_target(target_type: Any, target_id: Any) -> tuple[str, str]:
    normalized_target_type = str(target_type or "").strip().lower()
    if normalized_target_type not in {"plan", "policy"}:
        raise ValueError("Change request target_type must be `plan` or `policy`.")
    if normalized_target_type == "plan":
        normalized_target_id = _normalize_plan_name(target_id)
        if not normalized_target_id:
            raise ValueError("Change request target_id must not be empty for plans.")
        return normalized_target_type, normalized_target_id
    return normalized_target_type, normalize_tenant_id(str(target_id or ""))


def _normalize_change_request_action(value: Any) -> str:
    action = str(value or "upsert").strip().lower()
    if action not in {"upsert", "delete"}:
        raise ValueError("Change request action must be `upsert` or `delete`.")
    return action


def _default_change_request_approvals_required() -> int:
    return max(1, _env_int("ZOCR_API_TENANT_APPROVAL_MIN_APPROVERS", 1))


def _normalize_change_request_approvals_required(value: Any) -> int:
    if value is None:
        return _default_change_request_approvals_required()
    try:
        return max(1, int(value))
    except Exception as exc:
        raise RuntimeError("Change request approvals_required must be an integer >= 1.") from exc


def _normalize_change_request_approval_record(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Change request approval record must be an object.")
    reviewed_by = str(payload.get("reviewed_by") or "").strip()
    if not reviewed_by:
        raise RuntimeError("Change request approval record requires `reviewed_by`.")
    reviewed_at = payload.get("reviewed_at")
    if reviewed_at is not None and _parse_iso_datetime(reviewed_at) is None:
        raise RuntimeError("Invalid change request approval `reviewed_at` timestamp.")
    review_reason = str(payload.get("review_reason") or "").strip() or None
    return {
        "reviewed_by": reviewed_by,
        "reviewed_at": str(reviewed_at).strip() if reviewed_at is not None else None,
        "review_reason": review_reason,
    }


def _normalize_change_request_approval_records(value: Any) -> tuple[Dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("Change request approval_records must be an array.")
    seen: set[str] = set()
    records: list[Dict[str, Any]] = []
    for item in value:
        normalized = _normalize_change_request_approval_record(item)
        reviewer = str(normalized["reviewed_by"])
        if reviewer in seen:
            raise RuntimeError("Change request approval_records must not contain duplicate reviewers.")
        seen.add(reviewer)
        records.append(normalized)
    return tuple(records)


def _normalize_tenant_change_request_payload(payload: Any, *, request_id: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Tenant change request payload must be an object.")
    normalized_request_id = str(request_id or payload.get("request_id") or uuid.uuid4().hex).strip()
    if not normalized_request_id:
        raise RuntimeError("Tenant change request id must not be empty.")
    target_type, target_id = _normalize_change_request_target(
        payload.get("target_type"),
        payload.get("target_id"),
    )
    action = _normalize_change_request_action(payload.get("action"))
    requested_by = str(payload.get("requested_by") or "").strip() or None
    requested_at = payload.get("requested_at")
    if requested_at is not None and _parse_iso_datetime(requested_at) is None:
        raise RuntimeError("Invalid requested_at timestamp.")
    reviewed_by = str(payload.get("reviewed_by") or "").strip() or None
    reviewed_at = payload.get("reviewed_at")
    if reviewed_at is not None and _parse_iso_datetime(reviewed_at) is None:
        raise RuntimeError("Invalid reviewed_at timestamp.")
    status = _normalize_change_request_status(payload.get("status"))
    review_reason = str(payload.get("review_reason") or "").strip() or None
    approvals_required = _normalize_change_request_approvals_required(payload.get("approvals_required"))
    approval_records = _normalize_change_request_approval_records(payload.get("approval_records"))
    if len(approval_records) > approvals_required:
        raise RuntimeError("Change request approval_records exceed approvals_required.")
    raw_body = payload.get("payload")
    if raw_body is None:
        raw_body = {}
    if action == "upsert":
        if target_type == "plan":
            normalized_body = _normalize_tenant_plan_payload(target_id, raw_body)
        else:
            normalized_body = _normalize_tenant_policy_payload(target_id, raw_body)
    else:
        normalized_body = {}
    return {
        "request_id": normalized_request_id,
        "target_type": target_type,
        "target_id": target_id,
        "action": action,
        "payload": normalized_body,
        "status": status,
        "approvals_required": approvals_required,
        "approvals_received": len(approval_records),
        "approval_records": [dict(item) for item in approval_records],
        "requested_by": requested_by,
        "requested_at": str(requested_at).strip() if requested_at is not None else None,
        "reviewed_by": reviewed_by,
        "reviewed_at": str(reviewed_at).strip() if reviewed_at is not None else None,
        "review_reason": review_reason,
    }


def _tenant_change_request_from_payload(payload: Dict[str, Any], *, source: str) -> TenantChangeRequest:
    normalized = _normalize_tenant_change_request_payload(payload, request_id=payload.get("request_id"))
    return TenantChangeRequest(
        request_id=str(normalized["request_id"]),
        target_type=str(normalized["target_type"]),
        target_id=str(normalized["target_id"]),
        action=str(normalized["action"]),
        payload=dict(normalized["payload"]),
        status=str(normalized["status"]),
        approvals_required=int(normalized.get("approvals_required") or 1),
        approvals_received=int(normalized.get("approvals_received") or 0),
        approval_records=tuple(dict(item) for item in (normalized.get("approval_records") or [])),
        requested_by=normalized.get("requested_by"),
        requested_at=normalized.get("requested_at"),
        reviewed_by=normalized.get("reviewed_by"),
        reviewed_at=normalized.get("reviewed_at"),
        review_reason=normalized.get("review_reason"),
        source=source,
    )


def _self_approval_allowed() -> bool:
    return _env_truthy("ZOCR_API_TENANT_APPROVAL_ALLOW_SELF", False)


def _validated_change_request_reviewer(
    request: TenantChangeRequest,
    *,
    reviewed_by: str,
    action: str,
) -> str:
    reviewer = str(reviewed_by or "").strip()
    if not reviewer:
        raise ValueError("Reviewer identity must not be empty.")
    if (
        action == "approve"
        and not _self_approval_allowed()
        and request.requested_by
        and reviewer == request.requested_by
    ):
        raise PermissionError("Self-approval is not allowed.")
    return reviewer


def _append_change_request_approval(
    request: TenantChangeRequest,
    *,
    reviewed_by: str,
    review_reason: Optional[str],
) -> tuple[Dict[str, Any], ...]:
    if any(str(item.get("reviewed_by") or "") == reviewed_by for item in request.approval_records):
        raise ValueError("Reviewer already approved this change request.")
    return (
        *request.approval_records,
        {
            "reviewed_by": reviewed_by,
            "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "review_reason": str(review_reason or "").strip() or None,
        },
    )


class LocalJobRepository:
    backend = "local"

    def __init__(self, job_store: JobStore, *, default_tenant: str = "default"):
        self._job_store = job_store
        self._default_tenant = normalize_tenant_id(default_tenant or "default")
        self._env_tenant_plans = _load_tenant_plans_from_env()
        self._env_tenant_policies = _load_tenant_policies_from_env()
        self._config_dir = self._job_store.storage_dir / "_tenant_config"
        self._plans_path = self._config_dir / "tenant_plans.json"
        self._policies_path = self._config_dir / "tenant_policies.json"
        self._change_requests_path = self._config_dir / "tenant_change_requests.json"
        self._file_tenant_plans = self._load_json_file(self._plans_path, loader=_normalize_tenant_plan_payload)
        self._file_tenant_policies = self._load_json_file(
            self._policies_path,
            loader=_normalize_tenant_policy_payload,
        )
        self._file_change_requests = self._load_json_file(
            self._change_requests_path,
            loader=lambda key, value: _normalize_tenant_change_request_payload(value, request_id=str(key)),
        )

    def _payload_tenant(self, payload: Optional[Dict[str, Any]]) -> str:
        raw = payload.get("tenant_id") if isinstance(payload, dict) else None
        try:
            return normalize_tenant_id(str(raw or self._default_tenant))
        except ValueError:
            return self._default_tenant

    def _load_json_file(self, path: Path, *, loader) -> dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        items: dict[str, Dict[str, Any]] = {}
        if not isinstance(payload, dict):
            return items
        for key, value in payload.items():
            try:
                normalized = loader(str(key), value)
            except Exception:
                continue
            items[str(key)] = normalized
        return items

    def _write_json_file(self, path: Path, payload: dict[str, Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _merged_tenant_plans(self) -> dict[str, Dict[str, Any]]:
        merged = dict(self._env_tenant_plans)
        merged.update(self._file_tenant_plans)
        return merged

    def _merged_tenant_policies(self) -> dict[str, Dict[str, Any]]:
        merged = dict(self._env_tenant_policies)
        merged.update(self._file_tenant_policies)
        return merged

    def _apply_change_request(self, request: TenantChangeRequest) -> None:
        if request.target_type == "plan":
            if request.action == "upsert":
                self.upsert_tenant_plan(request.target_id, dict(request.payload))
                return
            self.delete_tenant_plan(request.target_id)
            return
        if request.action == "upsert":
            self.upsert_tenant_policy(request.target_id, dict(request.payload))
            return
        self.delete_tenant_policy(request.target_id)

    def _persist_change_request(self, payload: Dict[str, Any]) -> None:
        request_id = str(payload["request_id"])
        self._file_change_requests[request_id] = dict(payload)
        self._write_json_file(self._change_requests_path, self._file_change_requests)

    def list_job_refs(
        self,
        tenant_id: Optional[str] = None,
        *,
        limit: Optional[int] = None,
    ) -> list[JobRecordRef]:
        tenant_filter = normalize_tenant_id(tenant_id) if tenant_id else None
        items: list[JobRecordRef] = []
        for meta in self._job_store.list_job_metas():
            payload = self._job_store.read_job(meta.job_id)
            record_tenant = self._payload_tenant(payload)
            if tenant_filter and record_tenant != tenant_filter:
                continue
            items.append(
                JobRecordRef(
                    job_id=meta.job_id,
                    tenant_id=record_tenant,
                    last_modified=float(meta.last_modified),
                )
            )
        items.sort(key=lambda item: item.last_modified, reverse=True)
        if limit is not None:
            return items[: max(0, int(limit))]
        return items

    def read_job(self, job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        tenant_id = normalize_tenant_id(tenant_id)
        payload = self._job_store.read_job(job_id)
        if not isinstance(payload, dict):
            return None
        if self._payload_tenant(payload) != tenant_id:
            return None
        return payload

    def count_jobs(self, tenant_id: str, *, statuses: Optional[tuple[str, ...]] = None) -> int:
        tenant_id = normalize_tenant_id(tenant_id)
        status_filter = {str(item).strip() for item in (statuses or ()) if str(item).strip()}
        count = 0
        for meta in self.list_job_refs(tenant_id):
            payload = self._job_store.read_job(meta.job_id)
            if not isinstance(payload, dict):
                continue
            if self._payload_tenant(payload) != tenant_id:
                continue
            if status_filter and str(payload.get("status") or "").strip() not in status_filter:
                continue
            count += 1
        return count

    def list_tenant_plans(self) -> list[TenantPlan]:
        return [
            _tenant_plan_from_payload(plan_name, payload, source="local" if plan_name in self._file_tenant_plans else "env")
            for plan_name, payload in sorted(self._merged_tenant_plans().items())
        ]

    def upsert_tenant_plan(self, plan_name: str, payload: Dict[str, Any]) -> TenantPlan:
        normalized = _normalize_tenant_plan_payload(plan_name, payload)
        self._file_tenant_plans[str(normalized["plan_name"])] = normalized
        self._write_json_file(self._plans_path, self._file_tenant_plans)
        return _tenant_plan_from_payload(str(normalized["plan_name"]), normalized, source="local")

    def delete_tenant_plan(self, plan_name: str) -> bool:
        normalized_name = _normalize_plan_name(plan_name)
        if not normalized_name or normalized_name not in self._file_tenant_plans:
            return False
        self._file_tenant_plans.pop(normalized_name, None)
        self._write_json_file(self._plans_path, self._file_tenant_plans)
        return True

    def list_tenant_policies(self) -> list[TenantPolicy]:
        items: list[TenantPolicy] = []
        merged_plans = self._merged_tenant_plans()
        merged_policies = self._merged_tenant_policies()
        for tenant_id, payload in sorted(merged_policies.items()):
            plan_payload = None
            plan_name = _normalize_plan_name(payload.get("plan_name"))
            if plan_name:
                plan_payload = merged_plans.get(plan_name)
            resolved = _resolve_tenant_policy_record(
                tenant_id=tenant_id,
                source="local" if tenant_id in self._file_tenant_policies else "env",
                plan_payload=plan_payload,
                tenant_payload=payload,
            )
            if resolved is not None:
                items.append(resolved)
        return items

    def upsert_tenant_policy(self, tenant_id: str, payload: Dict[str, Any]) -> TenantPolicy:
        normalized = _normalize_tenant_policy_payload(tenant_id, payload)
        normalized_tenant = str(normalized["tenant_id"])
        self._file_tenant_policies[normalized_tenant] = normalized
        self._write_json_file(self._policies_path, self._file_tenant_policies)
        resolved = self.resolve_tenant_policy(normalized_tenant)
        if resolved is None:
            raise RuntimeError("Failed to persist tenant policy.")
        return resolved

    def delete_tenant_policy(self, tenant_id: str) -> bool:
        normalized_tenant = normalize_tenant_id(tenant_id)
        if normalized_tenant not in self._file_tenant_policies:
            return False
        self._file_tenant_policies.pop(normalized_tenant, None)
        self._write_json_file(self._policies_path, self._file_tenant_policies)
        return True

    def list_tenant_change_requests(
        self,
        *,
        status: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TenantChangeRequest]:
        normalized_status = _normalize_change_request_status(status) if status else None
        normalized_target_type = str(target_type or "").strip().lower() or None
        if normalized_target_type and normalized_target_type not in {"plan", "policy"}:
            raise ValueError("Change request target_type must be `plan` or `policy`.")
        items: list[TenantChangeRequest] = []
        for payload in self._file_change_requests.values():
            request = _tenant_change_request_from_payload(payload, source="local")
            if normalized_status and request.status != normalized_status:
                continue
            if normalized_target_type and request.target_type != normalized_target_type:
                continue
            items.append(request)
        items.sort(
            key=lambda item: (
                _parse_iso_datetime(item.requested_at) or datetime.fromtimestamp(0, tz=timezone.utc),
                item.request_id,
            ),
            reverse=True,
        )
        if limit is not None:
            return items[: max(0, int(limit))]
        return items

    def read_tenant_change_request(self, request_id: str) -> Optional[TenantChangeRequest]:
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            return None
        payload = self._file_change_requests.get(normalized_request_id)
        if not isinstance(payload, dict):
            return None
        return _tenant_change_request_from_payload(payload, source="local")

    def create_tenant_change_request(self, payload: Dict[str, Any]) -> TenantChangeRequest:
        normalized = _normalize_tenant_change_request_payload(
            {
                **dict(payload),
                "request_id": str(payload.get("request_id") or uuid.uuid4().hex),
                "status": "pending",
                "requested_at": payload.get("requested_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "reviewed_by": None,
                "reviewed_at": None,
                "review_reason": None,
            }
        )
        self._persist_change_request(normalized)
        return _tenant_change_request_from_payload(normalized, source="local")

    def approve_tenant_change_request(
        self,
        request_id: str,
        *,
        reviewed_by: str,
        review_reason: Optional[str] = None,
    ) -> Optional[TenantChangeRequest]:
        request = self.read_tenant_change_request(request_id)
        if request is None:
            return None
        if request.status != "pending":
            raise ValueError("Change request is not pending.")
        reviewer = _validated_change_request_reviewer(
            request,
            reviewed_by=reviewed_by,
            action="approve",
        )
        approval_records = _append_change_request_approval(
            request,
            reviewed_by=reviewer,
            review_reason=review_reason,
        )
        approved = len(approval_records) >= int(request.approvals_required)
        if approved:
            self._apply_change_request(request)
        normalized = _normalize_tenant_change_request_payload(
            {
                **request.__dict__,
                "status": "approved" if approved else "pending",
                "approval_records": [dict(item) for item in approval_records],
                "reviewed_by": reviewer if approved else None,
                "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat() if approved else None,
                "review_reason": review_reason if approved else None,
            },
            request_id=request.request_id,
        )
        self._persist_change_request(normalized)
        return _tenant_change_request_from_payload(normalized, source="local")

    def reject_tenant_change_request(
        self,
        request_id: str,
        *,
        reviewed_by: str,
        review_reason: Optional[str] = None,
    ) -> Optional[TenantChangeRequest]:
        request = self.read_tenant_change_request(request_id)
        if request is None:
            return None
        if request.status != "pending":
            raise ValueError("Change request is not pending.")
        reviewer = _validated_change_request_reviewer(
            request,
            reviewed_by=reviewed_by,
            action="reject",
        )
        if any(str(item.get("reviewed_by") or "") == reviewer for item in request.approval_records):
            raise ValueError("Reviewer already approved this change request.")
        normalized = _normalize_tenant_change_request_payload(
            {
                **request.__dict__,
                "status": "rejected",
                "reviewed_by": reviewer,
                "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "review_reason": review_reason,
            },
            request_id=request.request_id,
        )
        self._persist_change_request(normalized)
        return _tenant_change_request_from_payload(normalized, source="local")

    def resolve_tenant_policy(self, tenant_id: str) -> Optional[TenantPolicy]:
        tenant_id = normalize_tenant_id(tenant_id)
        tenant_payload = self._merged_tenant_policies().get(tenant_id)
        if not tenant_payload:
            return None
        plan_payload = None
        plan_name = _normalize_plan_name(tenant_payload.get("plan_name"))
        if plan_name:
            plan_payload = self._merged_tenant_plans().get(plan_name)
        return _resolve_tenant_policy_record(
            tenant_id=tenant_id,
            source="local" if tenant_id in self._file_tenant_policies else "env",
            plan_payload=plan_payload,
            tenant_payload=tenant_payload,
        )

    def write_job(self, job_id: str, tenant_id: str, payload: Dict[str, Any]) -> None:
        tenant_id = normalize_tenant_id(tenant_id)
        payload = dict(payload)
        payload["tenant_id"] = tenant_id
        self._job_store.write_job(job_id, payload)

    def delete_job(self, job_id: str, tenant_id: str) -> None:
        payload = self._job_store.read_job(job_id)
        if isinstance(payload, dict) and self._payload_tenant(payload) != normalize_tenant_id(tenant_id):
            return
        self._job_store.delete_job(job_id)


class PostgresJobRepository:
    backend = "postgres"

    def __init__(self, *, dsn: str, auto_init: bool = True):
        try:
            import psycopg  # type: ignore
            from psycopg.types.json import Jsonb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Postgres metadata requires `psycopg`. Install with `pip install -e '.[api_db]'`."
            ) from exc

        self._psycopg = psycopg
        self._jsonb = Jsonb
        self._dsn = (dsn or "").strip()
        self._auto_init = bool(auto_init)
        self._init_lock = Lock()
        self._initialized = False
        if not self._dsn:
            raise RuntimeError(
                "ZOCR_API_DATABASE_URL is required when ZOCR_API_METADATA_BACKEND=postgres."
            )

    def _connect(self):
        return self._psycopg.connect(self._dsn)

    def _maybe_init_schema(self) -> None:
        if not self._auto_init or self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._init_schema()
            self._initialized = True

    def _decode_payload(self, raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except Exception:
                return None
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    def _payload_timestamp(self, payload: Dict[str, Any], key: str) -> datetime:
        parsed = _parse_iso_datetime(payload.get(key))
        if parsed is not None:
            return parsed.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def _init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS zocr_jobs (
                tenant_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                payload JSONB NOT NULL,
                PRIMARY KEY (tenant_id, job_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS zocr_tenant_plans (
                plan_name TEXT PRIMARY KEY,
                max_active_jobs INTEGER,
                max_stored_jobs INTEGER,
                rate_limit_per_min INTEGER,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS zocr_tenant_policies (
                tenant_id TEXT PRIMARY KEY,
                plan_name TEXT REFERENCES zocr_tenant_plans(plan_name) ON DELETE SET NULL,
                max_active_jobs INTEGER,
                max_stored_jobs INTEGER,
                rate_limit_per_min INTEGER,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS zocr_tenant_change_requests (
                request_id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                approvals_required INTEGER NOT NULL DEFAULT 1,
                approval_records JSONB NOT NULL DEFAULT '[]'::jsonb,
                requested_by TEXT,
                requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reviewed_by TEXT,
                reviewed_at TIMESTAMPTZ,
                review_reason TEXT,
                payload JSONB NOT NULL
            )
            """,
            """
            ALTER TABLE zocr_tenant_change_requests
            ADD COLUMN IF NOT EXISTS approvals_required INTEGER NOT NULL DEFAULT 1
            """,
            """
            ALTER TABLE zocr_tenant_change_requests
            ADD COLUMN IF NOT EXISTS approval_records JSONB NOT NULL DEFAULT '[]'::jsonb
            """,
            "CREATE INDEX IF NOT EXISTS zocr_jobs_updated_at_idx ON zocr_jobs (updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS zocr_jobs_tenant_updated_idx ON zocr_jobs (tenant_id, updated_at DESC)",
            "CREATE INDEX IF NOT EXISTS zocr_jobs_tenant_status_idx ON zocr_jobs (tenant_id, status)",
            "CREATE INDEX IF NOT EXISTS zocr_tenant_policies_plan_idx ON zocr_tenant_policies (plan_name)",
            "CREATE INDEX IF NOT EXISTS zocr_tenant_change_requests_status_idx ON zocr_tenant_change_requests (status, requested_at DESC)",
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
            conn.commit()

    def list_job_refs(
        self,
        tenant_id: Optional[str] = None,
        *,
        limit: Optional[int] = None,
    ) -> list[JobRecordRef]:
        self._maybe_init_schema()
        params: list[Any] = []
        sql = "SELECT job_id, tenant_id, EXTRACT(EPOCH FROM updated_at) FROM zocr_jobs"
        if tenant_id:
            sql += " WHERE tenant_id = %s"
            params.append(normalize_tenant_id(tenant_id))
        sql += " ORDER BY updated_at DESC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(max(0, int(limit)))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        items = [
            JobRecordRef(job_id=str(job_id), tenant_id=str(tid), last_modified=float(last_modified or 0.0))
            for job_id, tid, last_modified in rows
        ]
        return items

    def read_job(self, job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        tenant_id = normalize_tenant_id(tenant_id)
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM zocr_jobs WHERE tenant_id = %s AND job_id = %s",
                    (tenant_id, job_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        return self._decode_payload(row[0])

    def count_jobs(self, tenant_id: str, *, statuses: Optional[tuple[str, ...]] = None) -> int:
        tenant_id = normalize_tenant_id(tenant_id)
        self._maybe_init_schema()
        sql = "SELECT COUNT(*) FROM zocr_jobs WHERE tenant_id = %s"
        params: list[Any] = [tenant_id]
        status_filter = [str(item).strip() for item in (statuses or ()) if str(item).strip()]
        if status_filter:
            sql += " AND status = ANY(%s)"
            params.append(status_filter)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
        return int(row[0] or 0) if row else 0

    def list_tenant_plans(self) -> list[TenantPlan]:
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT plan_name, max_active_jobs, max_stored_jobs, rate_limit_per_min
                    FROM zocr_tenant_plans
                    ORDER BY plan_name ASC
                    """
                )
                rows = cur.fetchall()
        return [
            TenantPlan(
                plan_name=str(plan_name),
                max_active_jobs=_coerce_optional_limit(max_active_jobs),
                max_stored_jobs=_coerce_optional_limit(max_stored_jobs),
                rate_limit_per_min=_coerce_optional_limit(rate_limit_per_min),
                source="postgres",
            )
            for plan_name, max_active_jobs, max_stored_jobs, rate_limit_per_min in rows
        ]

    def upsert_tenant_plan(self, plan_name: str, payload: Dict[str, Any]) -> TenantPlan:
        normalized = _normalize_tenant_plan_payload(plan_name, payload)
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO zocr_tenant_plans (
                        plan_name,
                        max_active_jobs,
                        max_stored_jobs,
                        rate_limit_per_min,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (plan_name)
                    DO UPDATE SET
                        max_active_jobs = EXCLUDED.max_active_jobs,
                        max_stored_jobs = EXCLUDED.max_stored_jobs,
                        rate_limit_per_min = EXCLUDED.rate_limit_per_min,
                        updated_at = NOW()
                    """,
                    (
                        normalized["plan_name"],
                        normalized.get("max_active_jobs"),
                        normalized.get("max_stored_jobs"),
                        normalized.get("rate_limit_per_min"),
                    ),
                )
            conn.commit()
        return _tenant_plan_from_payload(str(normalized["plan_name"]), normalized, source="postgres")

    def delete_tenant_plan(self, plan_name: str) -> bool:
        normalized_name = _normalize_plan_name(plan_name)
        if not normalized_name:
            return False
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM zocr_tenant_plans WHERE plan_name = %s", (normalized_name,))
                deleted = getattr(cur, "rowcount", 0)
            conn.commit()
        return bool(deleted)

    def list_tenant_policies(self) -> list[TenantPolicy]:
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        tp.tenant_id,
                        tp.plan_name,
                        COALESCE(tp.max_active_jobs, pl.max_active_jobs),
                        COALESCE(tp.max_stored_jobs, pl.max_stored_jobs),
                        COALESCE(tp.rate_limit_per_min, pl.rate_limit_per_min)
                    FROM zocr_tenant_policies tp
                    LEFT JOIN zocr_tenant_plans pl
                        ON pl.plan_name = tp.plan_name
                    ORDER BY tp.tenant_id ASC
                    """
                )
                rows = cur.fetchall()
        items: list[TenantPolicy] = []
        for tenant_id, plan_name, max_active_jobs, max_stored_jobs, rate_limit_per_min in rows:
            resolved = _resolve_tenant_policy_record(
                tenant_id=str(tenant_id),
                source="postgres",
                plan_payload=None,
                tenant_payload={
                    "plan_name": plan_name,
                    "max_active_jobs": max_active_jobs,
                    "max_stored_jobs": max_stored_jobs,
                    "rate_limit_per_min": rate_limit_per_min,
                },
            )
            if resolved is not None:
                items.append(resolved)
        return items

    def upsert_tenant_policy(self, tenant_id: str, payload: Dict[str, Any]) -> TenantPolicy:
        normalized = _normalize_tenant_policy_payload(tenant_id, payload)
        normalized_tenant = str(normalized["tenant_id"])
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO zocr_tenant_policies (
                        tenant_id,
                        plan_name,
                        max_active_jobs,
                        max_stored_jobs,
                        rate_limit_per_min,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (tenant_id)
                    DO UPDATE SET
                        plan_name = EXCLUDED.plan_name,
                        max_active_jobs = EXCLUDED.max_active_jobs,
                        max_stored_jobs = EXCLUDED.max_stored_jobs,
                        rate_limit_per_min = EXCLUDED.rate_limit_per_min,
                        updated_at = NOW()
                    """,
                    (
                        normalized_tenant,
                        normalized.get("plan_name"),
                        normalized.get("max_active_jobs"),
                        normalized.get("max_stored_jobs"),
                        normalized.get("rate_limit_per_min"),
                    ),
                )
            conn.commit()
        resolved = self.resolve_tenant_policy(normalized_tenant)
        if resolved is None:
            raise RuntimeError("Failed to persist tenant policy.")
        return resolved

    def delete_tenant_policy(self, tenant_id: str) -> bool:
        normalized_tenant = normalize_tenant_id(tenant_id)
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM zocr_tenant_policies WHERE tenant_id = %s", (normalized_tenant,))
                deleted = getattr(cur, "rowcount", 0)
            conn.commit()
        return bool(deleted)

    def list_tenant_change_requests(
        self,
        *,
        status: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TenantChangeRequest]:
        self._maybe_init_schema()
        normalized_status = _normalize_change_request_status(status) if status else None
        normalized_target_type = str(target_type or "").strip().lower() or None
        if normalized_target_type and normalized_target_type not in {"plan", "policy"}:
            raise ValueError("Change request target_type must be `plan` or `policy`.")
        sql = """
            SELECT
                request_id,
                target_type,
                target_id,
                action,
                status,
                approvals_required,
                approval_records,
                requested_by,
                requested_at,
                reviewed_by,
                reviewed_at,
                review_reason,
                payload
            FROM zocr_tenant_change_requests
        """
        clauses: list[str] = []
        params: list[Any] = []
        if normalized_status:
            clauses.append("status = %s")
            params.append(normalized_status)
        if normalized_target_type:
            clauses.append("target_type = %s")
            params.append(normalized_target_type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY requested_at DESC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(max(0, int(limit)))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        items: list[TenantChangeRequest] = []
        for (
            request_id,
            request_target_type,
            request_target_id,
            request_action,
            request_status,
            approvals_required,
            approval_records,
            requested_by,
            requested_at,
            reviewed_by,
            reviewed_at,
            review_reason,
            payload,
        ) in rows:
            items.append(
                _tenant_change_request_from_payload(
                    {
                        "request_id": request_id,
                        "target_type": request_target_type,
                        "target_id": request_target_id,
                        "action": request_action,
                        "status": request_status,
                        "approvals_required": approvals_required,
                        "approval_records": approval_records if isinstance(approval_records, list) else self._decode_payload(approval_records) or [],
                        "requested_by": requested_by,
                        "requested_at": requested_at.isoformat() if isinstance(requested_at, datetime) else requested_at,
                        "reviewed_by": reviewed_by,
                        "reviewed_at": reviewed_at.isoformat() if isinstance(reviewed_at, datetime) else reviewed_at,
                        "review_reason": review_reason,
                        "payload": payload if isinstance(payload, dict) else self._decode_payload(payload) or {},
                    },
                    source="postgres",
                )
            )
        return items

    def read_tenant_change_request(self, request_id: str) -> Optional[TenantChangeRequest]:
        normalized_request_id = str(request_id or "").strip()
        if not normalized_request_id:
            return None
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        request_id,
                        target_type,
                        target_id,
                        action,
                        status,
                        approvals_required,
                        approval_records,
                        requested_by,
                        requested_at,
                        reviewed_by,
                        reviewed_at,
                        review_reason,
                        payload
                    FROM zocr_tenant_change_requests
                    WHERE request_id = %s
                    """,
                    (normalized_request_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        (
            request_id,
            request_target_type,
            request_target_id,
            request_action,
            request_status,
            approvals_required,
            approval_records,
            requested_by,
            requested_at,
            reviewed_by,
            reviewed_at,
            review_reason,
            payload,
        ) = row
        return _tenant_change_request_from_payload(
            {
                "request_id": request_id,
                "target_type": request_target_type,
                "target_id": request_target_id,
                "action": request_action,
                "status": request_status,
                "approvals_required": approvals_required,
                "approval_records": approval_records if isinstance(approval_records, list) else self._decode_payload(approval_records) or [],
                "requested_by": requested_by,
                "requested_at": requested_at.isoformat() if isinstance(requested_at, datetime) else requested_at,
                "reviewed_by": reviewed_by,
                "reviewed_at": reviewed_at.isoformat() if isinstance(reviewed_at, datetime) else reviewed_at,
                "review_reason": review_reason,
                "payload": payload if isinstance(payload, dict) else self._decode_payload(payload) or {},
            },
            source="postgres",
        )

    def create_tenant_change_request(self, payload: Dict[str, Any]) -> TenantChangeRequest:
        normalized = _normalize_tenant_change_request_payload(
            {
                **dict(payload),
                "request_id": str(payload.get("request_id") or uuid.uuid4().hex),
                "status": "pending",
                "requested_at": payload.get("requested_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "reviewed_by": None,
                "reviewed_at": None,
                "review_reason": None,
            }
        )
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO zocr_tenant_change_requests (
                        request_id,
                        target_type,
                        target_id,
                        action,
                        status,
                        approvals_required,
                        approval_records,
                        requested_by,
                        requested_at,
                        reviewed_by,
                        reviewed_at,
                        review_reason,
                        payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        normalized["request_id"],
                        normalized["target_type"],
                        normalized["target_id"],
                        normalized["action"],
                        normalized["status"],
                        normalized.get("approvals_required"),
                        self._jsonb(normalized.get("approval_records") or []),
                        normalized.get("requested_by"),
                        self._payload_timestamp(normalized, "requested_at"),
                        normalized.get("reviewed_by"),
                        self._payload_timestamp(normalized, "reviewed_at") if normalized.get("reviewed_at") else None,
                        normalized.get("review_reason"),
                        self._jsonb(normalized["payload"]),
                    ),
                )
            conn.commit()
        created = self.read_tenant_change_request(str(normalized["request_id"]))
        if created is None:
            raise RuntimeError("Failed to persist tenant change request.")
        return created

    def approve_tenant_change_request(
        self,
        request_id: str,
        *,
        reviewed_by: str,
        review_reason: Optional[str] = None,
    ) -> Optional[TenantChangeRequest]:
        request = self.read_tenant_change_request(request_id)
        if request is None:
            return None
        if request.status != "pending":
            raise ValueError("Change request is not pending.")
        reviewer = _validated_change_request_reviewer(
            request,
            reviewed_by=reviewed_by,
            action="approve",
        )
        approval_records = _append_change_request_approval(
            request,
            reviewed_by=reviewer,
            review_reason=review_reason,
        )
        approved = len(approval_records) >= int(request.approvals_required)
        if approved:
            if request.target_type == "plan":
                if request.action == "upsert":
                    self.upsert_tenant_plan(request.target_id, dict(request.payload))
                else:
                    self.delete_tenant_plan(request.target_id)
            else:
                if request.action == "upsert":
                    self.upsert_tenant_policy(request.target_id, dict(request.payload))
                else:
                    self.delete_tenant_policy(request.target_id)
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE zocr_tenant_change_requests
                    SET
                        status = %s,
                        approval_records = %s,
                        reviewed_by = %s,
                        reviewed_at = %s,
                        review_reason = %s
                    WHERE request_id = %s
                    """,
                    (
                        "approved" if approved else "pending",
                        self._jsonb([dict(item) for item in approval_records]),
                        reviewer if approved else None,
                        datetime.now(timezone.utc) if approved else None,
                        review_reason if approved else None,
                        request.request_id,
                    ),
                )
            conn.commit()
        return self.read_tenant_change_request(request.request_id)

    def reject_tenant_change_request(
        self,
        request_id: str,
        *,
        reviewed_by: str,
        review_reason: Optional[str] = None,
    ) -> Optional[TenantChangeRequest]:
        request = self.read_tenant_change_request(request_id)
        if request is None:
            return None
        if request.status != "pending":
            raise ValueError("Change request is not pending.")
        reviewer = _validated_change_request_reviewer(
            request,
            reviewed_by=reviewed_by,
            action="reject",
        )
        if any(str(item.get("reviewed_by") or "") == reviewer for item in request.approval_records):
            raise ValueError("Reviewer already approved this change request.")
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE zocr_tenant_change_requests
                    SET
                        status = %s,
                        reviewed_by = %s,
                        reviewed_at = NOW(),
                        review_reason = %s,
                        approval_records = %s
                    WHERE request_id = %s
                    """,
                    (
                        "rejected",
                        reviewer,
                        review_reason,
                        self._jsonb([dict(item) for item in request.approval_records]),
                        request.request_id,
                    ),
                )
            conn.commit()
        return self.read_tenant_change_request(request.request_id)

    def resolve_tenant_policy(self, tenant_id: str) -> Optional[TenantPolicy]:
        tenant_id = normalize_tenant_id(tenant_id)
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        tp.plan_name,
                        COALESCE(tp.max_active_jobs, pl.max_active_jobs),
                        COALESCE(tp.max_stored_jobs, pl.max_stored_jobs),
                        COALESCE(tp.rate_limit_per_min, pl.rate_limit_per_min)
                    FROM zocr_tenant_policies tp
                    LEFT JOIN zocr_tenant_plans pl
                        ON pl.plan_name = tp.plan_name
                    WHERE tp.tenant_id = %s
                    """,
                    (tenant_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        plan_name, max_active_jobs, max_stored_jobs, rate_limit_per_min = row
        return _resolve_tenant_policy_record(
            tenant_id=tenant_id,
            source="postgres",
            plan_payload=None,
            tenant_payload={
                "plan_name": plan_name,
                "max_active_jobs": max_active_jobs,
                "max_stored_jobs": max_stored_jobs,
                "rate_limit_per_min": rate_limit_per_min,
            },
        )

    def write_job(self, job_id: str, tenant_id: str, payload: Dict[str, Any]) -> None:
        tenant_id = normalize_tenant_id(tenant_id)
        self._maybe_init_schema()
        payload = dict(payload)
        payload["tenant_id"] = tenant_id
        created_at = self._payload_timestamp(payload, "created_at")
        updated_at = self._payload_timestamp(payload, "updated_at")
        status = str(payload.get("status") or "queued")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO zocr_jobs (tenant_id, job_id, status, created_at, updated_at, payload)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, job_id)
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        payload = EXCLUDED.payload
                    """,
                    (
                        tenant_id,
                        job_id,
                        status,
                        created_at,
                        updated_at,
                        self._jsonb(payload),
                    ),
                )
            conn.commit()

    def delete_job(self, job_id: str, tenant_id: str) -> None:
        tenant_id = normalize_tenant_id(tenant_id)
        self._maybe_init_schema()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM zocr_jobs WHERE tenant_id = %s AND job_id = %s",
                    (tenant_id, job_id),
                )
            conn.commit()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def build_job_repository(job_store: JobStore) -> tuple[str, JobRepository]:
    backend = (os.environ.get("ZOCR_API_METADATA_BACKEND") or "local").strip().lower()
    default_tenant = default_tenant_from_env(
        multi_tenant_enabled=_env_truthy("ZOCR_API_MULTI_TENANT_ENABLED", False)
    )
    if backend in {"local", "filesystem", "fs"}:
        return "local", LocalJobRepository(job_store, default_tenant=default_tenant or "default")
    if backend in {"postgres", "postgresql", "pg"}:
        return "postgres", PostgresJobRepository(
            dsn=os.environ.get("ZOCR_API_DATABASE_URL") or "",
            auto_init=_env_truthy("ZOCR_API_DB_AUTO_INIT", True),
        )
    raise RuntimeError(
        "Unsupported ZOCR_API_METADATA_BACKEND=%r. Supported backends: local, postgres."
        % (backend,)
    )
