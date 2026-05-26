"""Main VLN instruction parser pipeline — compact output."""

from typing import Any, Dict, List, Optional

from .segmenter import (
    segment_instruction,
    normalize_whitespace,
    split_sentences_for_llm,
    has_cross_sentence_dependency,
)
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
            backtracking={},
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
            backtracking={},
            reason="vertical_motion_not_supported",
        )

    # Complex instructions must not be parsed by rules
    if requires_semantic_parser(cleaned_text):
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
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

    # Rule path: moderate confidence, no backtracking
    # If any task has UNKNOWN action, degrade to needs_review
    has_unknown = any(t.get("action") == "UNKNOWN" for t in tasks)
    result = _make_compact_result(
        status="needs_review" if has_unknown else "ok",
        confidence=0.85,
        tasks=tasks,
        constraints=[],
        backtracking={},
        reason="rule_fallback_due_to_llm_unavailable" if has_unknown else None,
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
    to receive uniform confidence/backtracking handling.

    Multi-sentence instructions are split sentence-by-sentence when no
    cross-sentence dependencies are detected; otherwise parsed as a whole
    with a needs_review marker.
    """
    cleaned_text = normalize_whitespace(text)

    # Pre-flight gates (MUST NOT depend on LLM)
    if not cleaned_text or not any(c.isalpha() and ord(c) < 128 for c in cleaned_text):
        result = _make_compact_result(
            status="ok",
            confidence=1.0,
            tasks=[],
            constraints=[],
            backtracking={},
        )
        validate_result(result)
        return result

    sentences = split_sentences_for_llm(cleaned_text)

    # Cross-sentence dependency check
    has_dep, dep_reason = has_cross_sentence_dependency(sentences)
    if has_dep:
        result = parse_instruction_llm(
            text,
            fallback_to_rules=True,
            vote_count=vote_count,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )
        # Force downgrade to needs_review because cross-sentence anaphora
        # makes per-sentence parsing unsafe
        result["status"] = "needs_review"
        result["reason"] = "cross_sentence_dependency_requires_review"
        if result.get("confidence", 0.0) > 0.9:
            result["confidence"] = 0.9
        return result

    # Single sentence: direct LLM parse
    if len(sentences) <= 1:
        return parse_instruction_llm(
            text,
            fallback_to_rules=True,
            vote_count=vote_count,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    # Multi-sentence without dependencies: parse each independently
    max_chunks = _max_sentence_chunks()
    if len(sentences) > max_chunks:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
            reason="sentence_chunk_limit_exceeded",
        )

    sentence_results: List[Dict[str, Any]] = []
    for sent in sentences:
        r = _parse_single_sentence_llm(
            sent,
            vote_count=vote_count,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )
        sentence_results.append(r)

    merged = _merge_sentence_results(sentence_results, cleaned_text)
    return merged


def _max_sentence_chunks() -> int:
    import os
    env_val = os.environ.get("VLN_MAX_SENTENCE_CHUNKS")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 8


def _parse_single_sentence_llm(
    text: str,
    vote_count: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Parse a single sentence via LLM vote → aggregate → verify → policy → backtracking.

    No pre-flight checks (caller already gated). No rule fallback.
    """
    from . import llm, aggregator

    success, raw_votes = llm.parse_with_llm(
        instruction=text,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

    if not success:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
            reason="sentence_chunk_unparsed",
        )

    # 1. Aggregate top-3 candidates for verifier
    candidates = aggregator.aggregate_votes(raw_votes, text)
    if not candidates:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
            reason="sentence_chunk_unparsed",
        )

    # 2. Collect ALL valid distinct plans for local step-candidate extraction
    all_valid_plans = aggregator.extract_valid_plan_pool(raw_votes)

    # 3. Verifier ranks top-3 candidates
    verifier_result = llm.verify_candidate_plans(
        text, candidates,
        base_url=base_url, model=model,
    )

    if verifier_result is not None:
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
            backtracking={},
            reason="confidence_verification_unavailable",
        )
        try:
            validate_result(result)
        except ValueError:
            return _make_compact_result(
                status="needs_review",
                confidence=0.0,
                tasks=[],
                constraints=[],
                backtracking={},
                reason="sentence_chunk_requires_review",
            )
        return result

    if not ranked_plans:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
            reason="sentence_chunk_unparsed",
        )

    # 4. Apply overall confidence policy (no alternatives)
    final = apply_plan_confidence_policy(ranked_plans)

    # 5. Extract local step candidates from all valid plans vs primary plan
    primary_plan = ranked_plans[0]
    step_pools = aggregator.extract_step_candidates(primary_plan, all_valid_plans)

    # 6. Step verifier: rate primary steps and candidates
    backtracking = {"step_candidates": []}
    if step_pools:
        step_verifier_result = llm.verify_step_candidates(
            text,
            primary_plan.get("tasks", []),
            step_pools,
            base_url=base_url, model=model,
        )
        if step_verifier_result is not None:
            backtracking = apply_step_candidate_policy(
                step_verifier_result.get("step_confidences", []),
                step_verifier_result.get("candidate_confidences", []),
                step_pools,
            )
        else:
            # Step verifier failed: keep empty backtracking, but if we had
            # candidates that looked close, degrade to needs_review
            if final.get("status") == "ok":
                final["status"] = "needs_review"
                final["reason"] = "step_confidence_verification_unavailable"

    final["backtracking"] = backtracking

    try:
        validate_result(final)
    except ValueError:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
            reason="sentence_chunk_requires_review",
        )

    return final


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
    4. Build step-level backtracking candidates
    5. Validate and return compact result
    """
    cleaned_text = normalize_whitespace(text)
    if not cleaned_text or not any(c.isalpha() and ord(c) < 128 for c in cleaned_text):
        result = _make_compact_result(
            status="ok",
            confidence=1.0,
            tasks=[],
            constraints=[],
            backtracking={},
        )
        validate_result(result)
        return result

    result = _parse_single_sentence_llm(
        text,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

    # Only fallback to rules when LLM produced no usable tasks.
    # A legitimate needs_review result with parsed tasks should NOT
    # be silently replaced by rule output, because that would lose candidates.
    if not result.get("tasks") and fallback_to_rules:
        return parse_instruction(text)

    return result


def apply_plan_confidence_policy(ranked_plans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply the three-tier confidence policy to choose the primary complete plan.

    This function NO LONGER generates full-plan alternatives.
    It only determines:
    - overall status (ok / needs_review / unsupported / none)
    - overall confidence
    - primary tasks and constraints

    C1 = first plan confidence
    C2 = second plan confidence
    C3 = third plan confidence

    Tier 1: C1 > 0.95 -> confidence=1.0, ok.
    Tier 2: 0.90 < C1 <= 0.95 -> ok if no non-localizable competition.
    Tier 3: C1 <= 0.90 -> needs_review.
    """
    if not ranked_plans:
        return _make_compact_result(
            status="none",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
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
        "backtracking": {},
    }
    if first.get("reason") is not None:
        base["reason"] = first["reason"]

    # Preserve unsupported/none status regardless of confidence
    if first_status in ("unsupported", "none"):
        return base

    def _diff(a, b):
        return round(a - b, 4)

    if c1 > 0.95:
        base["confidence"] = 1.0
        base["status"] = "ok"
        return base

    if c1 > 0.90:
        # Tier 2: check if close competitors can be localized to single steps.
        # If there are close full-plan competitors that differ in >1 step or
        # constraints, we mark needs_review because they cannot be expressed
        # as simple backtracking candidates.
        has_non_localizable = False
        if len(ranked_plans) >= 2:
            c2 = ranked_plans[1].get("confidence", 0.0)
            if _diff(c1, c2) < 0.15:
                # A close competitor exists at plan level.
                # For now, conservatively mark needs_review if we haven't
                # yet verified it can be localized (that happens later).
                has_non_localizable = True
        if len(ranked_plans) >= 3:
            c3 = ranked_plans[2].get("confidence", 0.0)
            if _diff(c1, c3) < 0.15:
                has_non_localizable = True
        base["status"] = "needs_review" if has_non_localizable else "ok"
    else:
        # Tier 3: always needs_review at plan level
        base["status"] = "needs_review"

    return base


def apply_step_candidate_policy(
    step_confidences: List[Dict[str, Any]],
    candidate_confidences: List[Dict[str, Any]],
    step_pools: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Apply per-step confidence policy to decide which candidates to keep
    in backtracking.step_candidates.

    Rules per step:
    - C1 > 0.95: no candidates saved
    - 0.90 < C1 <= 0.95: save candidates where C1 - Cn < 0.15
    - C1 <= 0.90: save candidates where C1 - Cn < 0.20
    - Max 2 candidates per step (rank 2, rank 3)

    Returns:
        {"step_candidates": [{"step_id": int, "candidates": [...]}]}
    """
    import json

    def _diff(a, b):
        return round(a - b, 4)

    # Map step_id -> primary confidence
    primary_conf: Dict[int, float] = {}
    for sc in step_confidences:
        sid = sc.get("step_id")
        if sid is not None:
            primary_conf[sid] = float(sc.get("confidence", 0.0))

    # Map (step_id, rank) -> candidate confidence
    cand_conf: Dict[tuple, float] = {}
    for cc in candidate_confidences:
        sid = cc.get("step_id")
        rank = cc.get("rank")
        if sid is not None and rank is not None:
            cand_conf[(sid, rank)] = float(cc.get("confidence", 0.0))

    step_candidates: List[Dict[str, Any]] = []

    for step_id in sorted(step_pools.keys()):
        c1 = primary_conf.get(step_id, 1.0)
        pool = step_pools[step_id]

        if c1 > 0.95:
            continue

        if c1 > 0.90:
            margin = 0.15
        else:
            margin = 0.20

        kept: List[Dict[str, Any]] = []
        for rank, cand_task in enumerate(pool, start=2):
            if rank > 3:
                break
            conf = cand_conf.get((step_id, rank), 0.0)
            if _diff(c1, conf) < margin:
                kept.append({
                    "rank": rank,
                    "step_id": step_id,
                    "action": cand_task.get("action", "UNKNOWN"),
                    "direction": cand_task.get("direction"),
                    "features": cand_task.get("features", []),
                    "confidence": conf,
                })

        if kept:
            step_candidates.append({
                "step_id": step_id,
                "candidates": kept,
            })

    return {"step_candidates": step_candidates}


def _merge_sentence_results(
    sentence_results: List[Dict[str, Any]],
    original_text: str,
) -> Dict[str, Any]:
    """
    Merge per-sentence compact results into a single multi-sentence compact result.

    Rules:
    - Status propagation: unsupported > needs_review > ok > none
    - Tasks: concatenated with renumbered step_ids
    - Constraints: union of all constraints
    - Confidence: minimum confidence across sentences (capped at 1.0)
    - Backtracking: renumber step_ids in step_candidates
    """
    if not sentence_results:
        return _make_compact_result(
            status="none",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={},
            reason="no_action",
        )

    # Status propagation priority
    statuses = [r.get("status", "none") for r in sentence_results]
    if any(s == "unsupported" for s in statuses):
        final_status = "unsupported"
        final_reason = "vertical_motion_not_supported"
    elif any(s == "needs_review" for s in statuses):
        final_status = "needs_review"
        reasons = []
        for r in sentence_results:
            if r.get("status") == "needs_review" and r.get("reason"):
                reasons.append(r["reason"])
        final_reason = reasons[0] if reasons else "sentence_chunk_requires_review"
    elif any(s == "ok" for s in statuses):
        final_status = "ok"
        final_reason = None
    else:
        final_status = "none"
        final_reason = "no_action"

    # Concatenate tasks with renumbered step_ids
    merged_tasks: List[Dict[str, Any]] = []
    offset = 0
    for r in sentence_results:
        tasks = r.get("tasks", [])
        for t in tasks:
            new_task = dict(t)
            new_task["step_id"] = t.get("step_id", 1) + offset
            merged_tasks.append(new_task)
        if tasks:
            offset += len(tasks)

    # Union constraints
    seen_constraints: set = set()
    merged_constraints: List[Dict[str, Any]] = []
    import json
    for r in sentence_results:
        for c in r.get("constraints", []):
            key = json.dumps(c, sort_keys=True)
            if key not in seen_constraints:
                seen_constraints.add(key)
                merged_constraints.append(c)

    # Confidence: minimum across sentences
    confidences = [r.get("confidence", 0.0) for r in sentence_results if r.get("tasks")]
    merged_confidence = min(confidences) if confidences else 0.0

    # For unsupported/none, tasks must be empty per validator
    if final_status in ("unsupported", "none"):
        merged_tasks = []

    # Merge backtracking: renumber step_ids
    merged_backtracking: Dict[str, Any] = {"step_candidates": []}
    offset = 0
    for r in sentence_results:
        tasks = r.get("tasks", [])
        bt = r.get("backtracking", {})
        for group in bt.get("step_candidates", []):
            new_group = {
                "step_id": group["step_id"] + offset,
                "candidates": [],
            }
            for cand in group.get("candidates", []):
                new_cand = dict(cand)
                new_cand["step_id"] = cand["step_id"] + offset
                new_group["candidates"].append(new_cand)
            merged_backtracking["step_candidates"].append(new_group)
        if tasks:
            offset += len(tasks)

    merged: Dict[str, Any] = {
        "status": final_status,
        "confidence": round(merged_confidence, 2),
        "tasks": merged_tasks,
        "constraints": merged_constraints,
        "backtracking": merged_backtracking,
    }
    if final_reason is not None:
        merged["reason"] = final_reason

    try:
        validate_result(merged)
    except ValueError:
        merged["status"] = "needs_review"
        merged["reason"] = "sentence_chunk_requires_review"
        merged["backtracking"] = {}

    return merged


def _make_compact_result(
    status: str,
    confidence: float,
    tasks: List[Dict[str, Any]],
    constraints: List[Dict[str, Any]],
    backtracking: Dict[str, Any],
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compact result dict."""
    result: Dict[str, Any] = {
        "status": status,
        "confidence": round(confidence, 2),
        "tasks": tasks,
        "constraints": constraints,
        "alternatives": [],
        "backtracking": backtracking,
    }
    if reason is not None:
        result["reason"] = reason
    return result
