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
            alternatives=[],
        )
        validate_result(result)
        return result

    from .complexity import has_active_vertical_motion
    if has_active_vertical_motion(cleaned_text):
        return _make_compact_result(
            status="unsupported",
            confidence=1.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="vertical_motion_not_supported",
        )

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
            alternatives=[],
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
    Parse a single sentence via LLM vote → aggregate → verify → policy.

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
            alternatives=[],
            reason="sentence_chunk_unparsed",
        )

    candidates = aggregator.aggregate_votes(raw_votes, text)
    if not candidates:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="sentence_chunk_unparsed",
        )

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
            alternatives=[],
            reason="confidence_verification_unavailable",
        )
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
            return _make_compact_result(
                status="needs_review",
                confidence=0.0,
                tasks=[],
                constraints=[],
                alternatives=[],
                reason="sentence_chunk_requires_review",
            )
        return result

    if not ranked_plans:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="sentence_chunk_unparsed",
        )

    final = apply_plan_confidence_policy(ranked_plans)

    try:
        validate_result(final)
    except ValueError:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
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
    4. Validate and return compact result
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

    result = _parse_single_sentence_llm(
        text,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

    if result["status"] in ("needs_review", "unsupported", "none") and fallback_to_rules:
        return parse_instruction(text)

    return result


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
    - Alternatives: generate full-plan alternatives by substituting one sentence's
      alternative at a time; max 2 full-plan alternatives (rank 2/3).
    """
    if not sentence_results:
        return _make_compact_result(
            status="none",
            confidence=0.0,
            tasks=[],
            constraints=[],
            alternatives=[],
            reason="no_action",
        )

    # Status propagation priority
    statuses = [r.get("status", "none") for r in sentence_results]
    if any(s == "unsupported" for s in statuses):
        final_status = "unsupported"
        final_reason = "vertical_motion_not_supported"
    elif any(s == "needs_review" for s in statuses):
        final_status = "needs_review"
        # Collect reasons from failing chunks
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

    # Build base merged result
    merged: Dict[str, Any] = {
        "status": final_status,
        "confidence": round(merged_confidence, 2),
        "tasks": merged_tasks,
        "constraints": merged_constraints,
        "alternatives": [],
    }
    if final_reason is not None:
        merged["reason"] = final_reason

    # Alternative generation: replace one sentence at a time
    # Gather (sentence_idx, alt) pairs from sentence-level alternatives
    alt_slots: List[tuple] = []
    for sent_idx, r in enumerate(sentence_results):
        for alt in r.get("alternatives", []):
            rank = alt.get("rank")
            if rank in (2, 3):
                alt_slots.append((sent_idx, rank, alt))

    # Sort by rank ascending (rank 2 before rank 3)
    alt_slots.sort(key=lambda x: x[1])

    # Build full-plan alternatives, replacing one sentence at a time
    full_plan_alts: List[Dict[str, Any]] = []
    used_ranks: set = set()
    for sent_idx, rank, alt in alt_slots:
        if rank in used_ranks:
            continue
        # Build full plan by substituting this sentence's alternative
        alt_tasks: List[Dict[str, Any]] = []
        alt_constraints: List[Dict[str, Any]] = []
        alt_confs: List[float] = []
        offset = 0
        for i, r in enumerate(sentence_results):
            if i == sent_idx:
                tasks = alt.get("tasks", [])
                conf = alt.get("confidence", 0.0)
            else:
                tasks = r.get("tasks", [])
                conf = r.get("confidence", 0.0)
            for t in tasks:
                new_task = dict(t)
                new_task["step_id"] = t.get("step_id", 1) + offset
                alt_tasks.append(new_task)
            if tasks:
                offset += len(tasks)
            if i == sent_idx:
                alt_constraints.extend(alt.get("constraints", []))
            else:
                alt_constraints.extend(r.get("constraints", []))
            alt_confs.append(conf)

        # Deduplicate constraints
        seen: set = set()
        deduped_constraints: List[Dict[str, Any]] = []
        for c in alt_constraints:
            key = json.dumps(c, sort_keys=True)
            if key not in seen:
                seen.add(key)
                deduped_constraints.append(c)

        full_plan_alts.append({
            "rank": rank,
            "confidence": min(alt_confs) if alt_confs else 0.0,
            "tasks": alt_tasks,
            "constraints": deduped_constraints,
        })
        used_ranks.add(rank)

    merged["alternatives"] = full_plan_alts[:2]

    # Validator bans alternatives on ok status; downgrade preemptively
    if merged["alternatives"] and merged["status"] == "ok":
        merged["status"] = "needs_review"

    try:
        validate_result(merged)
    except ValueError:
        # Degrade to needs_review if merged plan is invalid
        merged["status"] = "needs_review"
        merged["reason"] = "sentence_chunk_requires_review"
        merged["alternatives"] = []

    return merged


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
