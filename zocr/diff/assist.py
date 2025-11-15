# -*- coding: utf-8 -*-
"""Plan downstream assistance (reanalysis / RAG handoff) from diff events."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "invoice": [
        "invoice",
        "請求",
        "請求書",
        "facture",
        "subtotal",
        "tax",
        "消費税",
        "tva",
        "due",
        "payment",
        "支払",
        "vendor",
        "client",
        "顧客",
        "bank",
    ],
    "contract": [
        "contract",
        "agreement",
        "条項",
        "clause",
        "article",
        "term",
        "obligation",
        "覚書",
        "amendment",
    ],
    "procurement": [
        "purchase order",
        "po",
        "p.o",
        "発注",
        "納品",
        "supplier",
        "vendor",
        "rfq",
        "quote",
        "qty",
        "unit price",
    ],
    "logistics": [
        "shipment",
        "delivery",
        "ship",
        "輸送",
        "出荷",
        "waybill",
        "carrier",
        "port",
        "container",
    ],
    "finance": [
        "statement",
        "transaction",
        "balance",
        "deposit",
        "withdrawal",
        "credit",
        "debit",
        "残高",
        "入金",
        "出金",
    ],
    "legal": [
        "legal",
        "法務",
        "条文",
        "statute",
        "litigation",
        "clause",
        "compliance",
        "規約",
        "条項",
    ],
    "hr": [
        "payroll",
        "salary",
        "bonus",
        "employee",
        "従業員",
        "人事",
        "benefit",
        "compensation",
    ],
}

DOMAIN_SUMMARY_NOTES: Dict[str, str] = {
    "invoice": "Invoice / billing delta – validate totals, taxes, and due dates before notifying finance.",
    "contract": "Contract / clause drift – cite article/section numbers for legal assistants.",
    "procurement": "Purchase-order style change – highlight supplier, item rows, and delivery terms.",
    "logistics": "Logistics / shipment diff – include tracking IDs and quantity moves for ops teams.",
    "finance": "Banking / statement update – enumerate transaction IDs and balance impacts.",
    "legal": "Legal memorandum shift – mention clause titles and compliance owners.",
    "hr": "HR / payroll change – reference employee identifiers and pay-period context.",
    "general": "General structured document – keep row/table context and cite trace IDs for quick reopening.",
}

DOMAIN_DIRECTIVES: Dict[str, str] = {
    "invoice": "Emphasize totals, subtotals, tax codes, and vendor/customer identities.",
    "contract": "Point assistants to the clause labels and describe how obligations shifted.",
    "procurement": "Call out PO numbers, supplier names, and quantity/unit price changes.",
    "logistics": "Surface shipment IDs, container counts, and route/port movements.",
    "finance": "List affected transaction IDs and the before/after balances.",
    "legal": "Cite the article/title that moved and whether its level changed.",
    "hr": "Mention employee identifiers and pay-period spans for HR reruns.",
    "general": "Stick to the table context and include trace IDs so downstream LLMs can reopen the snippet.",
}

LLM_ACTION_GUIDANCE: Dict[str, str] = {
    "reanalyze_cells": "Re-run OCR / structured parsing on the referenced cells before promoting them downstream.",
    "rag_followup": "Prepare a concise explanation for the downstream RAG assistant describing what changed and why the stakeholder should care.",
    "profile_update": "Update extraction heuristics or header dictionaries so future runs stay aligned.",
}


@dataclass
class _Bucket:
    name: str
    entries: List[Dict[str, Any]]

    def add(self, entry: Dict[str, Any], limit: Optional[int]) -> None:
        if limit is not None and len(self.entries) >= limit:
            return
        self.entries.append(entry)


class DiffAssistPlanner:
    """Categorise diff events into reanalysis vs downstream RAG support tasks."""

    def __init__(
        self,
        reanalyze_similarity_threshold: float = 0.65,
        reanalyze_relative_delta: float = 0.05,
        numeric_abs_threshold: float = 1.0,
        max_items_per_bucket: Optional[int] = 200,
    ) -> None:
        self.reanalyze_similarity_threshold = reanalyze_similarity_threshold
        self.reanalyze_relative_delta = reanalyze_relative_delta
        self.numeric_abs_threshold = numeric_abs_threshold
        self.max_items_per_bucket = max_items_per_bucket

    def plan(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        reanalyze = _Bucket("reanalyze_queue", [])
        rag = _Bucket("rag_followups", [])
        profile = _Bucket("profile_actions", [])
        domain_counter: Dict[str, int] = defaultdict(int)

        def _bump(tags: List[str]) -> None:
            if not tags:
                return
            domain_counter[tags[0]] += 1

        for event in events:
            etype = event.get("type")
            if etype == "cell_updated":
                entry, severity = self._prepare_cell_entry(event)
                if not entry:
                    continue
                tags = self._annotate_entry(entry, event, action=entry["action"], severity=severity)
                _bump(tags)
                if severity == "high":
                    reanalyze.add(entry, self.max_items_per_bucket)
                else:
                    rag.add(entry, self.max_items_per_bucket)
            elif etype in {"row_added", "row_removed", "table_added", "table_removed", "section_added", "section_removed"}:
                entry = self._basic_entry(event, action="rag_followup", reason=self._row_or_table_reason(event))
                tags = self._annotate_entry(entry, event, action=entry["action"], severity=None)
                _bump(tags)
                rag.add(entry, self.max_items_per_bucket)
            elif etype in {"header_renamed", "col_moved", "section_title_changed", "section_level_changed"}:
                entry = self._basic_entry(event, action="profile_update", reason=self._profile_reason(event))
                tags = self._annotate_entry(entry, event, action=entry["action"], severity=None)
                _bump(tags)
                profile.add(entry, self.max_items_per_bucket)

        summary = {
            "events": len(events),
            "reanalyze_candidates": len(reanalyze.entries),
            "rag_followups": len(rag.entries),
            "profile_actions": len(profile.entries),
        }
        domain_briefings = self._domain_briefings(domain_counter)
        if domain_briefings:
            summary["primary_domain"] = domain_briefings[0]["domain"]
        elif domain_counter:
            summary["primary_domain"] = max(domain_counter.items(), key=lambda kv: kv[1])[0]
        return {
            "summary": summary,
            "reanalyze_queue": reanalyze.entries,
            "rag_followups": rag.entries,
            "profile_actions": profile.entries,
            "domain_briefings": domain_briefings,
        }

    # ------------------------------------------------------------------
    def _prepare_cell_entry(self, event: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        similarity = event.get("similarity")
        relative_delta = event.get("relative_delta")
        numeric_delta = event.get("numeric_delta")
        score = 0.0
        reason_parts: List[str] = []

        if similarity is None:
            score += 0.5
            reason_parts.append("text_missing")
        else:
            gap = max(0.0, 1.0 - float(similarity))
            score += gap
            reason_parts.append(f"similarity={similarity:.2f}")
        if relative_delta is not None:
            gap = min(1.0, abs(float(relative_delta)))
            score += gap
            reason_parts.append(f"relative_delta={relative_delta:+.2%}")
        elif numeric_delta is not None:
            gap = min(1.0, abs(float(numeric_delta)) / max(1.0, self.numeric_abs_threshold))
            score += gap
            reason_parts.append(f"numeric_delta={numeric_delta:+g}")

        if similarity is not None and similarity <= self.reanalyze_similarity_threshold:
            score = max(score, 1.0)
        if relative_delta is not None and abs(float(relative_delta)) >= self.reanalyze_relative_delta:
            score = max(score, 1.0)
        if numeric_delta is not None and abs(float(numeric_delta)) >= self.numeric_abs_threshold:
            score = max(score, 1.0)

        severity = "high" if score >= 1.0 else ("medium" if score >= 0.6 else "low")
        action = "reanalyze_cells" if severity == "high" else "rag_followup"
        reason = ", ".join(reason_parts) or "cell_changed"

        entry = {
            "action": action,
            "event_type": "cell_updated",
            "severity": severity,
            "reason": reason,
        }
        entry.update(self._extract_context(event))
        if severity == "low":
            # Low severity updates are usually noise; skip to keep plan concise
            return None, severity
        return entry, severity

    def _extract_context(self, event: Dict[str, Any]) -> Dict[str, Any]:
        context_keys = [
            "table_id",
            "table_page",
            "table_index",
            "table_rows",
            "table_columns",
            "table_headers",
            "table_header_preview",
            "row_key",
            "row_key_a",
            "row_key_b",
            "row_ids",
            "row_dates",
            "a_row_preview",
            "b_row_preview",
        ]
        context: Dict[str, Any] = {}
        for key in context_keys:
            if key in event and event[key] is not None:
                context[key] = event[key]
        if event.get("old") is not None:
            context["old"] = event.get("old")
        if event.get("new") is not None:
            context["new"] = event.get("new")
        if event.get("trace_a"):
            context["trace_a"] = event.get("trace_a")
        if event.get("trace_b"):
            context["trace_b"] = event.get("trace_b")
        if event.get("numeric_delta") is not None:
            context["numeric_delta"] = event.get("numeric_delta")
        if event.get("relative_delta") is not None:
            context["relative_delta"] = event.get("relative_delta")
        if event.get("similarity") is not None:
            context["similarity"] = event.get("similarity")
        return context

    def _basic_entry(self, event: Dict[str, Any], action: str, reason: str) -> Dict[str, Any]:
        payload = {"action": action, "event_type": event.get("type"), "reason": reason}
        payload.update(self._extract_context(event))
        if event.get("title"):
            payload["title"] = event.get("title")
        if event.get("level") is not None:
            payload["level"] = event.get("level")
        if event.get("page") is not None:
            payload["page"] = event.get("page")
        if event.get("from") is not None:
            payload["from"] = event.get("from")
        if event.get("to") is not None:
            payload["to"] = event.get("to")
        if event.get("from_level") is not None:
            payload["from_level"] = event.get("from_level")
        if event.get("to_level") is not None:
            payload["to_level"] = event.get("to_level")
        return payload

    def _row_or_table_reason(self, event: Dict[str, Any]) -> str:
        etype = event.get("type")
        if etype == "row_added":
            return "new row requires capture"
        if etype == "row_removed":
            return "row disappeared"
        if etype == "table_added":
            return "new table detected"
        if etype == "table_removed":
            return "table removed"
        if etype == "section_added":
            return "section added"
        if etype == "section_removed":
            return "section removed"
        return "structure changed"

    def _profile_reason(self, event: Dict[str, Any]) -> str:
        etype = event.get("type")
        if etype == "header_renamed":
            return f"header renamed {event.get('from')}→{event.get('to')}"
        if etype == "col_moved":
            return f"column moved {event.get('from_index')}→{event.get('to_index')}"
        if etype == "section_title_changed":
            return "section title changed"
        if etype == "section_level_changed":
            return "section level changed"
        return "profile adjustment"

    def _annotate_entry(
        self,
        entry: Dict[str, Any],
        event: Dict[str, Any],
        action: str,
        severity: Optional[str],
    ) -> List[str]:
        tags = self._infer_domains(event)
        if tags:
            entry["domain_tags"] = tags
        directive = self._llm_directive(action=action, event=event, domain_tags=tags, severity=severity)
        if directive:
            entry["llm_directive"] = directive
        return tags

    def _infer_domains(self, event: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []
        keys = [
            "table_id",
            "table_headers",
            "table_header_preview",
            "row_key",
            "row_key_a",
            "row_key_b",
            "row_ids",
            "row_dates",
            "title",
            "old",
            "new",
            "reason",
            "header",
            "from",
            "to",
        ]
        for key in keys:
            value = event.get(key)
            if isinstance(value, list):
                tokens.extend(str(v) for v in value if v)
            elif value:
                tokens.append(str(value))
        blob = " ".join(t.lower() for t in tokens if t).strip()
        if not blob:
            return ["general"]
        scores: List[Tuple[str, int]] = []
        for domain, patterns in DOMAIN_PATTERNS.items():
            score = sum(1 for pat in patterns if pat.lower() in blob)
            if score:
                scores.append((domain, score))
        if not scores:
            return ["general"]
        scores.sort(key=lambda kv: kv[1], reverse=True)
        ordered: List[str] = []
        for domain, _ in scores:
            if domain not in ordered:
                ordered.append(domain)
            if len(ordered) >= 3:
                break
        if "general" not in ordered:
            ordered.append("general")
        return ordered

    def _llm_directive(
        self,
        action: str,
        event: Dict[str, Any],
        domain_tags: List[str],
        severity: Optional[str],
    ) -> Optional[str]:
        action_hint = LLM_ACTION_GUIDANCE.get(action)
        if not action_hint:
            return None
        domain = domain_tags[0] if domain_tags else "general"
        snippet = DOMAIN_DIRECTIVES.get(domain, DOMAIN_DIRECTIVES["general"])
        details: List[str] = []
        row_ref = (
            event.get("row_key")
            or event.get("row_key_b")
            or event.get("row_key_a")
            or event.get("title")
        )
        if row_ref:
            details.append(f"row={row_ref}")
        if event.get("table_page") is not None:
            details.append(f"page={event['table_page']}")
        if event.get("table_index") is not None:
            details.append(f"table={event['table_index']}")
        header_preview = event.get("table_header_preview")
        if header_preview:
            details.append(f"headers={header_preview}")
        numeric_delta = event.get("numeric_delta")
        rel_delta = event.get("relative_delta")
        try:
            numeric_delta = float(numeric_delta) if numeric_delta is not None else None
        except (TypeError, ValueError):
            numeric_delta = None
        try:
            rel_delta = float(rel_delta) if rel_delta is not None else None
        except (TypeError, ValueError):
            rel_delta = None
        if numeric_delta is not None:
            delta_text = f"Δ={numeric_delta:+g}"
            if rel_delta is not None:
                delta_text += f" ({rel_delta:+.2%})"
            details.append(delta_text)
        elif rel_delta is not None:
            details.append(f"rΔ={rel_delta:+.2%}")
        old = event.get("old")
        new = event.get("new")
        if old is not None or new is not None:
            details.append(f"old='{old}' → new='{new}'")
        trace = event.get("trace_b") or event.get("trace_a")
        if trace:
            details.append(f"trace={trace}")
        if severity:
            details.append(f"severity={severity}")
        detail_text = " ".join(details).strip()
        return " ".join(
            part
            for part in [
                action_hint,
                f"Domain focus: {domain}.",
                snippet,
                detail_text,
            ]
            if part
        ).strip()

    def _domain_briefings(self, counter: Dict[str, int]) -> List[Dict[str, Any]]:
        if not counter:
            return []
        items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        briefings: List[Dict[str, Any]] = []
        for domain, count in items:
            if domain == "general":
                continue
            briefings.append(
                {
                    "domain": domain,
                    "events": count,
                    "note": DOMAIN_SUMMARY_NOTES.get(domain, DOMAIN_SUMMARY_NOTES["general"]),
                }
            )
        general_count = counter.get("general")
        if general_count:
            briefings.append(
                {
                    "domain": "general",
                    "events": general_count,
                    "note": DOMAIN_SUMMARY_NOTES["general"],
                }
            )
        return briefings[:5]
