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
    auto_accept_confidence: float = AUTO_ACCEPT_CONFIDENCE,
    auto_accept_margin: float = AUTO_ACCEPT_MARGIN,
    low_confidence_margin: float = LOW_CONFIDENCE_MARGIN,
) -> Dict[str, Any]:
    """
    Parse an English 2D VLN instruction into an ordered list of compact tasks (rule-based).

    Simple instructions only. If the instruction contains complex temporal or
    semantic structure, returns status="needs_review" with an empty task list.
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
            "confidence": 1.0,
        }
        if best.get("direction"):
            task["direction"] = best["direction"]

        tasks.append(task)
        context = update_context(context, task)

    # Simple rule path: always confidence 1.0, no alternatives
    result = _make_compact_result(
        status="ok",
        confidence=1.0,
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
    Recommended entry point: automatically chooses rule or LLM parser.

    Simple instructions -> rule-based parser.
    Complex instructions -> LLM parser with voting.
    """
    from .complexity import requires_semantic_parser
    if requires_semantic_parser(text):
        return parse_instruction_llm(
            text,
            fallback_to_rules=False,
            vote_count=vote_count,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )
    return parse_instruction(text)


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

    Returns a compact parse result dict.
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
        if fallback_to_rules and not requires_semantic_parser(text):
            result = parse_instruction(text)
            return result
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="llm_unavailable_for_complex_instruction",
        )

    agg_success, ranked_plans, needs_adjudication = aggregator.aggregate_votes(
        raw_votes, text
    )

    if needs_adjudication:
        candidates = aggregator.get_conflict_candidates(raw_votes)
        chosen_plan = llm.adjudicate_plan(
            text, candidates,
            base_url=base_url, model=model,
        )

        if chosen_plan is not None:
            # Build a synthetic draft from adjudicated plan and compile it
            draft = {
                "actions": [
                    {
                        "id": f"a{i+1}",
                        "action": step.get("action", "UNKNOWN"),
                        "direction": step.get("direction"),
                        "features": step.get("features", []),
                        "confidence": 0.85,
                    }
                    for i, step in enumerate(chosen_plan)
                ],
                "order": [],
                "constraints": [],
                "excluded": [],
            }
            compiled = compile_draft(draft)
            if compiled.get("status") != "unsupported":
                compiled["status"] = "needs_review"
                compiled["confidence"] = 0.89
                return compiled

    if not agg_success or not ranked_plans:
        if fallback_to_rules and not requires_semantic_parser(text):
            result = parse_instruction(text)
            return result
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="llm_aggregation_failed",
        )

    # Apply plan-level confidence policy
    final = apply_plan_confidence_policy(ranked_plans)

    try:
        validate_result(final)
    except ValueError:
        if fallback_to_rules and not requires_semantic_parser(text):
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

    if c1 > 0.90:
        # Tier 2
        if len(ranked_plans) >= 2:
            c2 = ranked_plans[1].get("confidence", 0.0)
            if c1 - c2 < 0.15:
                alternatives.append({
                    "rank": 2,
                    "confidence": c2,
                    "tasks": ranked_plans[1].get("tasks", []),
                    "constraints": ranked_plans[1].get("constraints", []),
                })
        if len(ranked_plans) >= 3:
            c3 = ranked_plans[2].get("confidence", 0.0)
            if c1 - c3 < 0.15:
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
            if c1 - c2 < 0.20:
                alternatives.append({
                    "rank": 2,
                    "confidence": c2,
                    "tasks": ranked_plans[1].get("tasks", []),
                    "constraints": ranked_plans[1].get("constraints", []),
                })
        if len(ranked_plans) >= 3:
            c3 = ranked_plans[2].get("confidence", 0.0)
            if c1 - c3 < 0.20:
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
