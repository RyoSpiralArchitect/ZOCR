# -*- coding: utf-8 -*-
"""Plan downstream assistance (reanalysis / RAG handoff) from diff events."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import count
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
    "insurance": [
        "policy",
        "claim",
        "premium",
        "coverage",
        "insurer",
        "insured",
        "deductible",
        "損害",
        "保険",
        "保険金",
        "保障",
    ],
    "healthcare": [
        "patient",
        "medical",
        "clinical",
        "診療",
        "薬価",
        "treatment",
        "provider",
        "hospital",
        "icd",
        "cpt",
    ],
    "manufacturing": [
        "bom",
        "bill of materials",
        "assembly",
        "sku",
        "part",
        "lot",
        "製造",
        "工程",
        "工番",
        "line",
    ],
    "energy": [
        "power",
        "utility",
        "generation",
        "grid",
        "kwh",
        "mw",
        "pipeline",
        "oil",
        "gas",
        "emission",
        "carbon",
    ],
    "compliance": [
        "audit",
        "compliance",
        "risk",
        "control",
        "penalty",
        "violation",
        "gdpr",
        "sox",
        "policy",
        "規制",
        "監査",
    ],
    "real_estate": [
        "lease",
        "rent",
        "tenant",
        "landlord",
        "property",
        "parcel",
        "sqft",
        "㎡",
        "建物",
        "土地",
        "assessor",
    ],
    "telecom": [
        "telecom",
        "carrier",
        "bandwidth",
        "5g",
        "lte",
        "sim",
        "通信",
        "network",
        "fiber",
        "spectrum",
    ],
    "retail": [
        "retail",
        "pos",
        "store",
        "店舗",
        "inventory",
        "sku",
        "merch",
        "category",
        "sales",
        "売上",
        "basket",
    ],
    "pharma": [
        "pharma",
        "drug",
        "compound",
        "clinical",
        "trial",
        "fda",
        "gmp",
        "dose",
        "治験",
        "薬",
        "製薬",
    ],
    "public_sector": [
        "municipal",
        "ordinance",
        "budget",
        "council",
        "prefecture",
        "省庁",
        "city",
        "公共",
        "procurement",
        "grant",
        "appropriation",
    ],
    "education": [
        "curriculum",
        "syllabus",
        "semester",
        "credit",
        "student",
        "enrollment",
        "学期",
        "授業",
        "faculty",
        "campus",
    ],
    "technology": [
        "release notes",
        "version",
        "api",
        "module",
        "repository",
        "commit",
        "feature",
        "テック",
        "システム",
        "仕様",
    ],
    "marketing": [
        "campaign",
        "impression",
        "ctr",
        "leads",
        "pipeline",
        "広告",
        "branding",
        "creative",
        "媒体",
        "budget",
    ],
    "aviation": [
        "flight",
        "aircraft",
        "tail",
        "iata",
        "icao",
        "runway",
        "gate",
        "crew",
        "airline",
        "航班",
        "機材",
    ],
    "construction": [
        "bid",
        "blueprint",
        "contractor",
        "site",
        "施工",
        "建設",
        "permit",
        "rfi",
        "punch list",
        "工程表",
    ],
    "automotive": [
        "vehicle",
        "fleet",
        "vin",
        "dealership",
        "odometer",
        "engine",
        "warranty",
        "車両",
        "車検",
        "走行",
    ],
    "hospitality": [
        "hotel",
        "resort",
        "room",
        "booking",
        "check-in",
        "adr",
        "revpar",
        "banquet",
        "宿泊",
        "旅館",
    ],
    "media": [
        "broadcast",
        "stream",
        "episode",
        "ratings",
        "viewership",
        "ad slot",
        "network",
        "媒体",
        "番組",
        "配信",
    ],
    "banking": [
        "loan",
        "credit line",
        "mortgage",
        "interest",
        "treasury",
        "basel",
        "branch",
        "貸出",
        "融資",
        "金利",
    ],
    "gaming": [
        "game",
        "quest",
        "dlc",
        "loot",
        "matchmaking",
        "esports",
        "server",
        "プレイ",
        "課金",
        "大会",
    ],
    "food_beverage": [
        "menu",
        "ingredient",
        "recipe",
        "brew",
        "batch",
        "restaurant",
        "cafe",
        "食品",
        "飲料",
        "酒類",
    ],
    "agriculture": [
        "crop",
        "harvest",
        "acre",
        "hectare",
        "irrigation",
        "soil",
        "yield",
        "農地",
        "農薬",
        "畜産",
    ],
    "mining": [
        "mine",
        "ore",
        "pit",
        "drill",
        "rig",
        "extraction",
        "shaft",
        "鉱山",
        "採掘",
        "選鉱",
    ],
    "shipping": [
        "bill of lading",
        "imo",
        "vessel",
        "voyage",
        "charter",
        "manifest",
        "港湾",
        "船積",
        "港",
        "航海",
    ],
    "sports": [
        "league",
        "season",
        "match",
        "score",
        "stadium",
        "ticket",
        "athlete",
        "試合",
        "得点",
        "チーム",
    ],
    "entertainment": [
        "studio",
        "script",
        "scene",
        "talent",
        "royalty",
        "licensing",
        "festival",
        "上映",
        "配給",
        "興行",
    ],
    "nonprofit": [
        "donation",
        "grant",
        "fundraising",
        "charity",
        "ngo",
        "npo",
        "mission",
        "寄付",
        "助成",
        "非営利",
    ],
    "cybersecurity": [
        "threat",
        "vulnerability",
        "patch",
        "incident",
        "breach",
        "cve",
        "cvss",
        "攻撃",
        "脆弱性",
        "マルウェア",
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
    "insurance": "Insurance policy or claim drift – call out policy numbers, coverage caps, and payout amounts.",
    "healthcare": "Healthcare/clinical diff – cite patient IDs, treatment codes, and physician sign-offs.",
    "manufacturing": "Manufacturing/BOM change – list affected part numbers, assembly steps, and lot IDs.",
    "energy": "Energy/utility adjustment – highlight meter IDs, generation totals, and emission factors.",
    "compliance": "Compliance/audit variance – mention control IDs, audit steps, and deadlines.",
    "real_estate": "Real-estate or lease shift – cite parcel IDs, lease numbers, and rent adjustments.",
    "telecom": "Telecom/network delta – call out carrier IDs, circuit numbers, and bandwidth impacts.",
    "retail": "Retail/POS change – list store IDs, SKU counts, and sales deltas per channel.",
    "pharma": "Pharma/clinical update – mention protocol IDs, compound names, and dose changes.",
    "public_sector": "Public-sector memo – reference ordinance numbers, budget lines, and grant codes.",
    "education": "Education/academic diff – cite course codes, semesters, and enrollment counts.",
    "technology": "Technology/spec update – highlight version numbers, API endpoints, and module owners.",
    "marketing": "Marketing/campaign shift – include campaign IDs, spend deltas, and KPI impacts.",
    "aviation": "Aviation/operations change – mention flight numbers, aircraft tails, and route timing.",
    "construction": "Construction/project variance – reference bid IDs, site codes, and schedule impacts.",
    "automotive": "Automotive/fleet adjustment – cite VINs, fleet IDs, service intervals, and mileage deltas.",
    "hospitality": "Hospitality/room block diff – include booking IDs, ADR/RevPAR shifts, and guest segments.",
    "media": "Media/broadcast update – point to episode IDs, rating points, and ad-slot changes.",
    "banking": "Banking/loan packet – mention facility IDs, rates, maturities, and covenant references.",
    "gaming": "Gaming/live-ops change – cite event names, DLC identifiers, and reward tables.",
    "food_beverage": "Food & beverage tweak – list menu items, batch/lot codes, and ingredient shifts.",
    "agriculture": "Agriculture/field diff – reference crop names, acreage, irrigation cycles, and yield deltas.",
    "mining": "Mining/production change – cite pit IDs, ore grades, and tonnage movements.",
    "shipping": "Marine/shipping update – include vessel/voyage IDs, B/L numbers, and port rotations.",
    "sports": "Sports/fixture change – mention league/match IDs, scores, and venue adjustments.",
    "entertainment": "Entertainment/content shift – highlight title IDs, talent commitments, and royalty impacts.",
    "nonprofit": "Nonprofit/fund memo – cite campaign IDs, grant references, and donor tiers.",
    "cybersecurity": "Security incident diff – list CVE/CVSS data, affected assets, and mitigation steps.",
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
    "insurance": "Reference policy numbers, claim IDs, coverage limits, and payout figures.",
    "healthcare": "Point to patient IDs, encounter dates, and treatment/procedure codes.",
    "manufacturing": "Surface BOM lines, part numbers, quantities, and assembly stage changes.",
    "energy": "List meter IDs, generation/consumption deltas, and emission metrics.",
    "compliance": "Tie the change back to control IDs, regulatory clauses, and due dates.",
    "real_estate": "Flag lease IDs, parcel references, and rent/tax adjustments.",
    "telecom": "Call out circuit IDs, bandwidth tiers, and outage windows.",
    "retail": "Mention store/SKU codes, unit deltas, and promo windows.",
    "pharma": "Reference protocol IDs, batch numbers, and dosing/timeline deltas.",
    "public_sector": "Point to ordinance numbers, budget line items, and grant identifiers.",
    "education": "List course codes, term spans, and enrollment counts.",
    "technology": "Highlight version numbers, API routes, and affected modules.",
    "marketing": "Mention campaign IDs, spend deltas, KPIs, and creative slots.",
    "aviation": "List flight numbers, aircraft tails, gate/runway changes, and timing deltas.",
    "construction": "Reference project/site codes, bid packages, and schedule/cost impacts.",
    "automotive": "Provide VINs, fleet IDs, service history, and warranty buckets for each change.",
    "hospitality": "Call out reservation IDs, check-in/check-out dates, ADR/RevPAR metrics, and guest types.",
    "media": "Reference program IDs, airing windows, ratings, and monetisation slots impacted.",
    "banking": "List facility IDs, principal balances, rates, and amortisation/covenant notes.",
    "gaming": "Mention event names, DLC/build numbers, reward odds, and live-ops scheduling.",
    "food_beverage": "Tie differences to menu SKUs, batch numbers, ingredient substitutions, and QA notes.",
    "agriculture": "Highlight crop/field identifiers, acreage, irrigation schedules, and forecast yields.",
    "mining": "Provide pit/shaft IDs, ore grades, extraction tonnage, and safety/environment tags.",
    "shipping": "Note vessel/voyage numbers, B/L references, container counts, and port sequences.",
    "sports": "Reference league/match IDs, home/away info, rosters, and ticket/attendance stats.",
    "entertainment": "List title IDs, script/scene references, talent/royalty clauses, and release windows.",
    "nonprofit": "Mention campaign/grant IDs, donor tiers, budget buckets, and compliance strings.",
    "cybersecurity": "Surface asset IDs, CVE/CVSS info, detection timestamps, and containment steps.",
    "general": "Stick to the table context and include trace IDs so downstream LLMs can reopen the snippet.",
}

LLM_ACTION_GUIDANCE: Dict[str, str] = {
    "reanalyze_cells": "Re-run OCR / structured parsing on the referenced cells before promoting them downstream.",
    "rag_followup": "Prepare a concise explanation for the downstream RAG assistant describing what changed and why the stakeholder should care.",
    "profile_update": "Update extraction heuristics or header dictionaries so future runs stay aligned.",
}

ACTION_CONTEXT_HINTS: Dict[str, str] = {
    "reanalyze_cells": "Re-run OCR/structure capture on the cited cells before escalating.",
    "rag_followup": "Draft a downstream explanation that highlights the contextual change.",
    "profile_update": "Update header/column/profile rules so future runs stay aligned.",
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
        seq = count(1)

        def _bump(tags: List[str]) -> None:
            if not tags:
                return
            domain_counter[tags[0]] += 1

        def _assign(entry: Dict[str, Any]) -> Dict[str, Any]:
            entry["entry_id"] = f"entry_{next(seq)}"
            return entry

        for event in events:
            etype = event.get("type")
            if etype == "cell_updated":
                entry, severity = self._prepare_cell_entry(event)
                if not entry:
                    continue
                tags = self._annotate_entry(entry, event, action=entry["action"], severity=severity)
                self._attach_llm_context(entry, tags, severity)
                _bump(tags)
                if severity == "high":
                    reanalyze.add(_assign(entry), self.max_items_per_bucket)
                else:
                    rag.add(_assign(entry), self.max_items_per_bucket)
            elif etype in {"row_added", "row_removed", "table_added", "table_removed", "section_added", "section_removed"}:
                entry = self._basic_entry(event, action="rag_followup", reason=self._row_or_table_reason(event))
                tags = self._annotate_entry(entry, event, action=entry["action"], severity=None)
                self._attach_llm_context(entry, tags, severity=None)
                _bump(tags)
                rag.add(_assign(entry), self.max_items_per_bucket)
            elif etype in {"header_renamed", "col_moved", "section_title_changed", "section_level_changed"}:
                entry = self._basic_entry(event, action="profile_update", reason=self._profile_reason(event))
                tags = self._annotate_entry(entry, event, action=entry["action"], severity=None)
                self._attach_llm_context(entry, tags, severity=None)
                _bump(tags)
                profile.add(_assign(entry), self.max_items_per_bucket)

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
        packets = self._build_handoff_packets([reanalyze, rag, profile])
        agentic_requests = self._build_agentic_requests(events)
        summary["agentic_requests"] = len(agentic_requests)
        return {
            "summary": summary,
            "reanalyze_queue": reanalyze.entries,
            "rag_followups": rag.entries,
            "profile_actions": profile.entries,
            "domain_briefings": domain_briefings,
            "handoff_packets": packets,
            "agentic_requests": agentic_requests,
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
            "a_row_context",
            "b_row_context",
            "row_context_radius",
            "section_heading",
            "section_level",
            "section_path",
            "section_heading_a",
            "section_level_a",
            "section_path_a",
            "section_heading_b",
            "section_level_b",
            "section_path_b",
            "line_signature",
            "line_label",
            "line_pair_cost",
            "line_pair_penalty",
            "line_pair_gap",
            "line_pair_status",
            "text_change_type",
            "text_token_stats",
            "text_highlight",
            "numeric_unit",
            "numeric_currency",
            "numeric_is_percent",
            "numeric_scale",
            "source",
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

    def _attach_llm_context(
        self,
        entry: Dict[str, Any],
        domain_tags: List[str],
        severity: Optional[str],
    ) -> None:
        context = self._llm_context(entry, domain_tags, severity)
        if context:
            entry["llm_ready_context"] = context
        brief = self._handoff_brief(entry, domain_tags, severity)
        if brief:
            entry["handoff_brief"] = brief

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

    def _llm_context(
        self,
        entry: Dict[str, Any],
        domain_tags: List[str],
        severity: Optional[str],
    ) -> Optional[str]:
        action = entry.get("action", "rag_followup")
        domain = domain_tags[0] if domain_tags else "general"
        severity_label = severity or entry.get("severity") or "info"
        lines: List[str] = []
        action_hint = ACTION_CONTEXT_HINTS.get(action)
        if action_hint:
            lines.append(action_hint)
        lines.append(
            f"Action={action} | Domain={domain} | Event={entry.get('event_type')} | Severity={severity_label}"
        )
        if entry.get("reason"):
            lines.append(f"Reason: {entry['reason']}")
        focus_parts: List[str] = []
        row_ref = entry.get("row_key") or entry.get("row_key_b") or entry.get("row_key_a")
        if row_ref:
            focus_parts.append(f"row={row_ref}")
        if entry.get("row_ids"):
            focus_parts.append("ids=" + ",".join(map(str, entry["row_ids"][:3])))
        if entry.get("row_dates"):
            focus_parts.append("dates=" + ",".join(map(str, entry["row_dates"][:3])))
        if entry.get("table_page") is not None:
            focus_parts.append(f"page={entry['table_page']}")
        if entry.get("table_index") is not None:
            focus_parts.append(f"table={entry['table_index']}")
        header_preview = entry.get("table_header_preview") or entry.get("table_headers")
        if header_preview:
            focus_parts.append(f"headers={header_preview}")
        if entry.get("title"):
            focus_parts.append(f"title={entry['title']}")
        if focus_parts:
            lines.append("Focus: " + ", ".join(focus_parts))
        value_parts: List[str] = []
        if entry.get("old") is not None or entry.get("new") is not None:
            value_parts.append(f"old='{entry.get('old')}' → new='{entry.get('new')}'")
        if entry.get("numeric_delta") is not None:
            value_parts.append(f"Δ={entry['numeric_delta']:+g}")
        if entry.get("relative_delta") is not None:
            try:
                value_parts.append(f"rΔ={float(entry['relative_delta']):+.2%}")
            except (TypeError, ValueError):
                pass
        if entry.get("similarity") is not None:
            value_parts.append(f"sim={float(entry['similarity']):.2f}")
        trace = entry.get("trace_b") or entry.get("trace_a")
        if trace:
            value_parts.append(f"trace={trace}")
        preview = entry.get("a_row_preview") or entry.get("b_row_preview") or entry.get("row_preview")
        if preview:
            value_parts.append(f"row≈{preview}")
        if value_parts:
            lines.append("Values: " + "; ".join(value_parts))
        return "\n".join(lines).strip() if lines else None

    def _handoff_brief(
        self,
        entry: Dict[str, Any],
        domain_tags: List[str],
        severity: Optional[str],
    ) -> Optional[str]:
        domain = domain_tags[0] if domain_tags else "general"
        action = entry.get("action", "rag_followup")
        severity_label = severity or entry.get("severity")
        parts = [f"{domain}:{entry.get('event_type')}→{action}"]
        if severity_label:
            parts.append(f"severity={severity_label}")
        if entry.get("reason"):
            parts.append(entry["reason"])
        focus = entry.get("row_key") or entry.get("row_key_b") or entry.get("title")
        if focus:
            parts.append(f"focus={focus}")
        if entry.get("numeric_delta") is not None:
            parts.append(f"Δ={entry['numeric_delta']:+g}")
        if entry.get("relative_delta") is not None:
            try:
                parts.append(f"rΔ={float(entry['relative_delta']):+.2%}")
            except (TypeError, ValueError):
                pass
        trace = entry.get("trace_b") or entry.get("trace_a")
        if trace:
            parts.append(f"trace={trace}")
        return " | ".join(part for part in parts if part)

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

    def _build_handoff_packets(self, buckets: List[_Bucket]) -> List[Dict[str, Any]]:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for bucket in buckets:
            for entry in bucket.entries:
                domain = (entry.get("domain_tags") or ["general"])[0]
                groups[(domain, entry.get("action", "rag_followup"))].append(entry)

        packets: List[Dict[str, Any]] = []
        for (domain, action), entries in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1])):
            summary_note = DOMAIN_SUMMARY_NOTES.get(domain, DOMAIN_SUMMARY_NOTES["general"])
            action_hint = LLM_ACTION_GUIDANCE.get(action, "")
            sample_refs = self._sample_entry_refs(entries)
            llm_prompt = self._packet_prompt(domain, action, entries, summary_note, action_hint, sample_refs)
            context_examples = [
                entry.get("llm_ready_context")
                for entry in entries[:3]
                if entry.get("llm_ready_context")
            ]
            brief_examples = [
                entry.get("handoff_brief")
                for entry in entries[:4]
                if entry.get("handoff_brief")
            ]
            packets.append(
                {
                    "domain": domain,
                    "action": action,
                    "count": len(entries),
                    "summary": summary_note,
                    "action_hint": action_hint,
                    "sample_refs": sample_refs,
                    "llm_prompt": llm_prompt,
                    "llm_context_examples": context_examples,
                    "handoff_briefs": brief_examples,
                    "entry_ids": [entry.get("entry_id") for entry in entries],
                }
            )
        return packets[:20]

    def _build_agentic_requests(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        seq = count(1)
        for event in events:
            ctx = self._extract_context(event)
            directive = self._agentic_directive(event, ctx)
            if not directive:
                continue
            domain_tags = self._infer_domains(event)
            request = {
                "request_id": f"agentic_{next(seq)}",
                "event_type": event.get("type"),
                "domain_tags": domain_tags,
                "llm_directive": directive,
                "preferred_outputs": self._agentic_outputs(event),
                "context": ctx,
                "source": event.get("source") or "semantic_diff",
            }
            focus = self._agentic_focus(event, ctx)
            if focus:
                request["handoff_focus"] = focus
            visual = self._agentic_visual_brief(event, ctx)
            if visual:
                request["visual_brief"] = visual
            narrative = self._agentic_narrative_brief(event, ctx, domain_tags)
            if narrative:
                request["narrative_brief"] = narrative
            if event.get("text_highlight"):
                request["annotated_excerpt"] = event.get("text_highlight")
            requests.append(request)
        return requests

    def _agentic_outputs(self, event: Dict[str, Any]) -> List[str]:
        outputs = ["narrative_explanation"]
        etype = event.get("type")
        if etype in {"cell_updated", "row_added", "row_removed", "table_added", "table_removed"}:
            outputs.append("diff_overlay_image")
        if event.get("text_change_type"):
            outputs.append("annotated_excerpt")
        if event.get("section_heading") or event.get("section_heading_a") or event.get("section_heading_b"):
            outputs.append("section_summary_card")
        return outputs

    def _agentic_focus(self, event: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[List[str]]:
        focus: List[str] = []
        row_label = ctx.get("row_key") or ctx.get("line_label")
        heading = ctx.get("section_heading") or ctx.get("section_heading_b") or ctx.get("section_heading_a")
        if row_label:
            focus.append(f"Emphasize row/line '{row_label}'.")
        if heading:
            focus.append(f"Mention section '{heading}'.")
        if event.get("type") == "cell_updated" and ctx.get("old") is not None and ctx.get("new") is not None:
            focus.append("Show before vs after numeric values in a badge or overlay.")
        if event.get("text_change_type"):
            focus.append("Highlight the rewritten sentence and explain why it changed.")
        if not focus:
            return None
        return focus

    def _agentic_visual_brief(self, event: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[str]:
        row_label = ctx.get("row_key") or ctx.get("line_label")
        old = ctx.get("old")
        new = ctx.get("new")
        if event.get("type") == "cell_updated" and row_label and old is not None and new is not None:
            return (
                f"Render a miniature table strip for '{row_label}' showing the old value '{old}'"
                f" versus the new value '{new}' with arrows/colour chips."
            )
        if event.get("text_change_type") and ctx.get("text_highlight"):
            return "Create a side-by-side paragraph snippet with additions/removals highlighted."
        if event.get("type") in {"row_added", "row_removed"} and row_label:
            return f"Show the full row '{row_label}' and annotate whether it was added or removed."
        return None

    def _agentic_narrative_brief(
        self,
        event: Dict[str, Any],
        ctx: Dict[str, Any],
        domain_tags: List[str],
    ) -> Optional[str]:
        domain = domain_tags[0] if domain_tags else "general"
        row_label = ctx.get("row_key") or ctx.get("line_label") or "this entry"
        heading = ctx.get("section_heading") or ctx.get("section_path") or ctx.get("section_heading_b")
        etype = event.get("type")
        if etype == "cell_updated":
            old = ctx.get("old")
            new = ctx.get("new")
            if old is None and new is None:
                return None
            return (
                f"Explain in JA/EN how {row_label} under {heading or 'the noted section'} changed from"
                f" '{old}' to '{new}', and why that matters for the {domain} workflow."
            )
        if event.get("text_change_type"):
            change_type = event.get("text_change_type")
            return (
                f"Summarize the {change_type} in {row_label} within {heading or 'the current section'} and"
                " clarify the intent (policy note, legal clause, etc.) for downstream RAG agents."
            )
        if etype in {"row_added", "row_removed"}:
            action = "was added" if etype == "row_added" else "was removed"
            return f"Describe why row '{row_label}' {action} and suggest any follow-up actions for {domain} reviewers."
        if etype in {"table_added", "table_removed"}:
            action = "appeared" if etype == "table_added" else "disappeared"
            return f"Document that a table {action} and list the primary headers so the Agentic RAG can create visual summaries."
        return None

    def _agentic_directive(self, event: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[str]:
        etype = event.get("type")
        row_label = ctx.get("row_key") or ctx.get("line_label") or ctx.get("row_key_b")
        heading = ctx.get("section_heading") or ctx.get("section_path") or ctx.get("section_heading_b")
        old = ctx.get("old")
        new = ctx.get("new")
        base = [
            "Produce a concise explanation in Japanese *and* English so end users understand the change without reopening the document.",
            "If tools allow, prepare a diff-style image or overlay plus optional narration for the Agentic RAG GUI.",
            "Retain key identifiers (row keys, trace IDs, section titles) in your response for downstream linking.",
        ]
        if etype == "cell_updated" and old is not None and new is not None:
            detail = f"Focus on {row_label or 'the affected row'} (section {heading or '?'}) changing from '{old}' to '{new}'."
        elif event.get("text_change_type"):
            change_type = event.get("text_change_type")
            detail = (
                f"Describe the {change_type} text update around {row_label or 'the paragraph'}"
                f" and highlight the edits shown in `text_highlight`."
            )
        elif etype in {"row_added", "row_removed"}:
            action = "addition" if etype == "row_added" else "removal"
            detail = f"Explain the {action} of row {row_label or ''} and note any downstream tasks."
        elif etype in {"table_added", "table_removed"}:
            detail = "Summarize why the table structure changed and what users should verify."
        else:
            detail = "Summarize the structural adjustment for downstream review."
        base.append(detail)
        return " ".join(base)

    def _sample_entry_refs(self, entries: List[Dict[str, Any]], limit: int = 8) -> List[str]:
        refs: List[str] = []
        for entry in entries:
            ref = (
                entry.get("row_key")
                or entry.get("row_key_b")
                or entry.get("row_key_a")
                or entry.get("title")
                or entry.get("table_header_preview")
                or entry.get("reason")
            )
            if ref:
                refs.append(str(ref))
            if len(refs) >= limit:
                break
        return refs

    def _packet_prompt(
        self,
        domain: str,
        action: str,
        entries: List[Dict[str, Any]],
        summary_note: str,
        action_hint: str,
        sample_refs: List[str],
    ) -> str:
        bullet_lines: List[str] = []
        for entry in entries[: min(5, len(entries))]:
            ref = entry.get("row_key") or entry.get("row_key_b") or entry.get("title") or entry.get("reason")
            delta = []
            if entry.get("numeric_delta") is not None:
                delta.append(f"Δ={entry['numeric_delta']:+g}")
            if entry.get("relative_delta") is not None:
                try:
                    delta.append(f"rΔ={float(entry['relative_delta']):+.2%}")
                except (TypeError, ValueError):
                    pass
            if entry.get("reason"):
                delta.append(entry["reason"])
            delta_text = ", ".join(delta)
            bullet_lines.append(f"- {ref or entry.get('event_type')} {delta_text}".strip())
        sample_blob = "; ".join(sample_refs) if sample_refs else ""
        parts = [
            f"Domain: {domain} ({summary_note})",
            f"Action: {action}.",
            action_hint,
            "Focus rows/sections: " + sample_blob if sample_blob else "",
            "Key deltas:\n" + "\n".join(bullet_lines) if bullet_lines else "",
        ]
        return "\n".join(part for part in parts if part).strip()
