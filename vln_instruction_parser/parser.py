"""Main VLN instruction parser pipeline — compact output."""

from typing import Any, Dict, List, Optional

from .segmenter import segment_instruction, normalize_whitespace
from .extractor import extract_candidates
from .resolver import resolve_context_for_candidates, update_context
from .validator import validate_result
from .semantic_compiler import compile_draft

AUTO_ACCEPT_CONFIDENCE = 0.90
AUTO_ACCEPT_MARGIN = 0.15
LOW_CONFIDENCE_MARGIN = 0.20


def parse_instruction(
    text: str,
) -> Dict[str, Any]:
    """
    Explicit rule-based parser. Returns compact tasks for simple instructions.

    This is a limited fallback interface. Results are NOT automatically accepted
    as high-confidence execution plans.
    """
    cleaned_text = normalize_whitespace(text)

    # English-only heuristic
    if not cleaned_text or not any(c.isalpha() and ord(c) < 128 for c in cleaned_text):
        result = _make_compact_result(
            status="ok",
            confidence=1.0,
            tasks=[],
            constraints=[],
            alternatives=[],
        )
        validate_result(result)
        return result

    # Active vertical motion is unsupported in 2D
    from .complexity import has_active_vertical_motion, requires_semantic_parser
    if has_active_vertical_motion(cleaned_text):
        return _make_compact_result(
            status="unsupported",
            confidence=1.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="vertical_motion_not_supported",
        )

    # Complex instructions must not be parsed by rules
    if requires_semantic_parser(cleaned_text):
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="semantic_parser_required",
        )

    segments = segment_instruction(cleaned_text)

    tasks = []
    context = {}

    for step_id, segment in enumerate(segments, start=1):
        candidates = extract_candidates(segment, context)
        candidates = resolve_context_for_candidates(candidates, context)
        candidates = sorted(candidates, key=lambda c: c["confidence"], reverse=True)

        best = candidates[0]

        task = {
            "step_id": step_id,
            "action": best["action"],
            "features": best.get("features", []),
            "confidence": 0.85,
        }
        if best.get("direction"):
            task["direction"] = best["direction"]

        tasks.append(task)
        context = update_context(context, task)

    # Rule path: moderate confidence, no alternatives
    result = _make_compact_result(
        status="ok",
        confidence=0.85,
        tasks=tasks,
        constraints=[],
        alternatives=[],
    )

    validate_result(result)
    return result


def parse_instruction_auto(
    text: str,
    vote_count: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Recommended entry point: LLM-first semantic pipeline for all valid English 2D instructions.

    Simple and complex instructions both go through the LLM semantic pipeline
    to receive uniform confidence/alternatives handling.
    """
    cleaned_text = normalize_whitespace(text)
    if not cleaned_text or not any(c.isalpha() and ord(c) < 128 for c in cleaned_text):
        result = _make_compact_result(
            status="ok",
            confidence=1.0,
            tasks=[],
            constraints=[],
            alternatives=[],
        )
        validate_result(result)
        return result

    return parse_instruction_llm(
        text,
        fallback_to_rules=True,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def parse_instruction_llm(
    text: str,
    fallback_to_rules: bool = True,
    vote_count: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Parse an English 2D VLN instruction using an LLM with semantic execution-order understanding.

    Flow:
    1. Vote -> compile drafts -> group into top-3 candidates
    2. Verifier ranks candidates with plan-level confidence
    3. Apply confidence policy with 0.95/0.90 thresholds
    4. Validate and return compact result
    """
    from . import llm, aggregator
    from .complexity import requires_semantic_parser

    cleaned_text = normalize_whitespace(text)
    if not cleaned_text or not any(c.isalpha() and ord(c) < 128 for c in cleaned_text):
        result = _make_compact_result(
            status="ok",
            confidence=1.0,
            tasks=[],
            constraints=[],
            alternatives=[],
        )
        validate_result(result)
        return result

    success, raw_votes = llm.parse_with_llm(
        instruction=text,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

    if not success:
        if fallback_to_rules:
            return parse_instruction(text)
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="llm_unavailable_for_complex_instruction",
        )

    # Step 1: aggregate votes into top-3 compact candidates
    candidates = aggregator.aggregate_votes(raw_votes, text)

    if not candidates:
        if fallback_to_rules:
            return parse_instruction(text)
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="llm_aggregation_failed",
        )

    # Step 2: verifier reranker
    verifier_result = llm.verify_candidate_plans(
        text, candidates,
        base_url=base_url, model=model,
    )

    if verifier_result is not None:
        # Map verifier confidence back to candidates
        ranked_plans = []
        for item in verifier_result:
            cid = item["candidate_id"]
            conf = item["confidence"]
            cand = next((c for c in candidates if c.get("candidate_id") == cid), None)
            if cand is not None:
                plan = {
                    "status": cand.get("status", "ok"),
                    "confidence": conf,
                    "tasks": cand.get("tasks", []),
                    "constraints": cand.get("constraints", []),
                }
                if cand.get("reason") is not None:
                    plan["reason"] = cand["reason"]
                ranked_plans.append(plan)
    else:
        # Verifier failed: conservative degradation
        main = candidates[0]
        result = _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=main.get("tasks", []),
            constraints=main.get("constraints", []),
            alternatives=[],
            reason="confidence_verification_unavailable",
        )
        # Add up to two remaining candidates as alternatives with confidence 0.0
        for idx, alt in enumerate(candidates[1:3], start=2):
            result["alternatives"].append({
                "rank": idx,
                "confidence": 0.0,
                "tasks": alt.get("tasks", []),
                "constraints": alt.get("constraints", []),
            })
        try:
            validate_result(result)
        except ValueError:
            if fallback_to_rules:
                return parse_instruction(text)
            return _make_compact_result(
                status="needs_review",
                confidence=0.0,
                tasks=[],
                constraints=[],
                alternatives=[],
                reason="llm_validation_failed",
            )
        return result

    if not ranked_plans:
        if fallback_to_rules:
            return parse_instruction(text)
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="llm_aggregation_failed",
        )

    # Step 3: apply confidence policy
    final = apply_plan_confidence_policy(ranked_plans)

    try:
        validate_result(final)
    except ValueError:
        if fallback_to_rules:
            return parse_instruction(text)
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="llm_validation_failed",
        )

    return final


def apply_plan_confidence_policy(ranked_plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply the three-tier confidence policy to a list of ranked compact plans.

    C1 = first plan confidence
    C2 = second plan confidence
    C3 = third plan confidence

    Tier 1: C1 > 0.95 -> confidence=1.0, ok, no alternatives.
    Tier 2: 0.90 < C1 <= 0.95 -> compare C1-C2 < 0.15, C1-C3 < 0.15.
    Tier 3: C1 <= 0.90 -> compare C1-C2 < 0.20, C1-C3 < 0.20.
    """
    if not ranked_plans:
        return _make_compact_result(
            status="none",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="no_action",
        )

    first = ranked_plans[0]
    c1 = first.get("confidence", 0.0)
    first_status = first.get("status", "ok")

    # Build base result from first plan
    base: Dict[str, Any] = {
        "status": first_status,
        "confidence": c1,
        "tasks": first.get("tasks", []),
        "constraints": first.get("constraints", []),
        "alternatives": [],
    }
    if first.get("reason") is not None:
        base["reason"] = first["reason"]

    # Preserve unsupported/none status regardless of confidence
    if first_status in ("unsupported", "none"):
        return base

    if c1 > 0.95:
        base["confidence"] = 1.0
        base["status"] = "ok"
        return base

    alternatives: List[Dict[str, Any]] = []

    def _diff(a, b):
        return round(a - b, 4)

    if c1 > 0.90:
        # Tier 2
        if len(ranked_plans) >= 2:
            c2 = ranked_plans[1].get("confidence", 0.0)
            if _diff(c1, c2) < 0.15:
                alternatives.append({
                    "rank": 2,
                    "confidence": c2,
                    "tasks": ranked_plans[1].get("tasks", []),
                    "constraints": ranked_plans[1].get("constraints", []),
                })
        if len(ranked_plans) >= 3:
            c3 = ranked_plans[2].get("confidence", 0.0)
            if _diff(c1, c3) < 0.15:
                alternatives.append({
                    "rank": 3,
                    "confidence": c3,
                    "tasks": ranked_plans[2].get("tasks", []),
                    "constraints": ranked_plans[2].get("constraints", []),
                })
        base["status"] = "ok" if not alternatives else "needs_review"
    else:
        # Tier 3
        if len(ranked_plans) >= 2:
            c2 = ranked_plans[1].get("confidence", 0.0)
            if _diff(c1, c2) < 0.20:
                alternatives.append({
                    "rank": 2,
                    "confidence": c2,
                    "tasks": ranked_plans[1].get("tasks", []),
                    "constraints": ranked_plans[1].get("constraints", []),
                })
        if len(ranked_plans) >= 3:
            c3 = ranked_plans[2].get("confidence", 0.0)
            if _diff(c1, c3) < 0.20:
                alternatives.append({
                    "rank": 3,
                    "confidence": c3,
                    "tasks": ranked_plans[2].get("tasks", []),
                    "constraints": ranked_plans[2].get("constraints", []),
                })
        base["status"] = "needs_review"

    base["alternatives"] = alternatives
    return base


def _make_compact_result(
    status: str,
    confidence: float,
    tasks: List[Dict[str, Any]],
    constraints: List[Dict[str, Any]],
    alternatives: List[Dict[str, Any]],
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compact result dict."""
    result: Dict[str, Any] = {
        "status": status,
        "confidence": round(confidence, 2),
        "tasks": tasks,
        "constraints": constraints,
        "alternatives": alternatives,
    }
    if reason is not None:
        result["reason"] = reason
    return result
