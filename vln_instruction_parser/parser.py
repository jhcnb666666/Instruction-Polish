"""Main VLN instruction parser pipeline — compact output."""

import os
import re
from typing import Any, Dict, List, Optional

from .segmenter import (
    segment_instruction,
    normalize_whitespace,
    split_sentences_for_llm,
)
from .extractor import extract_candidates
from .resolver import resolve_context_for_candidates, update_context
from .validator import validate_result
from .semantic_compiler import compile_draft

AUTO_ACCEPT_CONFIDENCE = 0.90
AUTO_ACCEPT_MARGIN = 0.15
LOW_CONFIDENCE_MARGIN = 0.20

MAX_SEGMENT_SENTENCES = 5
DEFAULT_SEGMENT_MAX_PHASES = 5
DEFAULT_SEGMENT_MAX_WORDS = 120
_FIDELITY_BLOCKING_REASONS = {
    "missing_action",
    "extra_action",
    "wrong_order",
    "wrong_direction",
    "wrong_landmark",
    "missing_condition",
    "wrong_constraint",
    "negation_lost",
    "unsupported_motion",
    "unresolved_reference",
    "invalid_step_mapping",
}


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

    Instructions within the sentence, navigation-phase, and word budgets are
    parsed as a whole. Longer instruction chains are split by an LLM into
    ordered contiguous semantic segments. Each segment runs the full short-
    instruction pipeline before the merged candidate plans are audited against
    the complete instruction.
    """
    cleaned_text = normalize_whitespace(text)

    # Pre-flight gates (MUST NOT depend on LLM)
    if not cleaned_text or not any(c.isalpha() and ord(c) < 128 for c in cleaned_text):
        result = _make_compact_result(
            status="ok",
            confidence=1.0,
            tasks=[],
            constraints=[],
            backtracking={"step_candidates": []},
        )
        validate_result(result)
        return result

    max_sentences = _sentence_split_threshold()
    max_phases = _segment_max_phases()
    max_words = _segment_max_words()

    # Short path: unified full-instruction parse
    if not _exceeds_segment_budget(
        cleaned_text,
        max_sentences=max_sentences,
        max_phases=max_phases,
        max_words=max_words,
    ):
        return parse_instruction_llm(
            text,
            fallback_to_rules=True,
            vote_count=1 if vote_count is None else vote_count,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    # Long path: semantic segments, each using the complete short pipeline.
    from . import llm
    from .complexity import has_active_vertical_motion

    if has_active_vertical_motion(cleaned_text):
        return _make_compact_result(
            status="unsupported",
            confidence=1.0,
            tasks=[],
            constraints=[],
            backtracking={"step_candidates": []},
            reason="vertical_motion_not_supported",
        )

    segments = llm.split_instruction_semantically(
        cleaned_text,
        max_sentences=max_sentences,
        max_phases=max_phases,
        max_words=max_words,
        base_url=base_url,
        model=model,
    )
    if not _valid_semantic_segments(
        cleaned_text,
        segments,
        max_sentences=max_sentences,
        max_phases=max_phases,
        max_words=max_words,
    ):
        return _parse_whole_instruction(
            text, vote_count, base_url, model, temperature
        )

    segment_results: List[Dict[str, Any]] = []
    for segment in segments or []:
        result = parse_instruction_llm(
            segment,
            fallback_to_rules=True,
            vote_count=1 if vote_count is None else vote_count,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )
        if result.get("status") == "unsupported":
            return result
        if not result.get("tasks") and not result.get("constraints"):
            return _parse_whole_instruction(
                text, vote_count, base_url, model, temperature
            )
        segment_results.append(result)

    plans = _merge_segment_results_into_plans(segment_results)
    if not plans:
        return _parse_whole_instruction(
            text, vote_count, base_url, model, temperature
        )

    # Run global postprocess on the merged plan against the full original instruction
    # and any local step alternatives retained by the segment pipelines.
    merged_result = _run_global_postprocess(
        text, plans,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
        retry_audit=True,
        preserve_tasks_on_audit_failure=True,
    )
    if _merged_result_requires_whole_parse(merged_result):
        return _parse_whole_instruction(
            text, vote_count, base_url, model, temperature
        )
    return merged_result


def _sentence_split_threshold() -> int:
    """Return a configurable segment sentence limit, never exceeding five."""
    env_val = os.environ.get("VLN_SENTENCE_SPLIT_THRESHOLD")
    parsed = _parse_positive_int(env_val)
    if parsed is not None:
        return min(parsed, MAX_SEGMENT_SENTENCES)
    # Backward compatibility alias
    env_val = os.environ.get("VLN_MAX_SENTENCE_CHUNKS")
    parsed = _parse_positive_int(env_val)
    if parsed is not None:
        return min(parsed, MAX_SEGMENT_SENTENCES)
    return MAX_SEGMENT_SENTENCES


def _segment_max_phases() -> int:
    """Return the maximum recognized navigation phases allowed per segment."""
    parsed = _parse_positive_int(os.environ.get("VLN_SEGMENT_MAX_PHASES"))
    if parsed is None:
        return DEFAULT_SEGMENT_MAX_PHASES
    return min(parsed, DEFAULT_SEGMENT_MAX_PHASES)


def _segment_max_words() -> int:
    """Return the maximum normalized English word count allowed per segment."""
    parsed = _parse_positive_int(os.environ.get("VLN_SEGMENT_MAX_WORDS"))
    if parsed is None:
        return DEFAULT_SEGMENT_MAX_WORDS
    return min(parsed, DEFAULT_SEGMENT_MAX_WORDS)


def _parse_positive_int(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(value, 1)


def _instruction_word_count(text: str) -> int:
    """Count English words for LLM-segment size control."""
    return len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text))


def _exceeds_segment_budget(
    text: str,
    max_sentences: int,
    max_phases: int,
    max_words: int,
) -> bool:
    """Return True when an instruction needs semantic segmentation."""
    from .complexity import count_navigation_phases

    return (
        len(split_sentences_for_llm(text)) > max_sentences
        or count_navigation_phases(text) > max_phases
        or _instruction_word_count(text) > max_words
    )


def _parse_whole_instruction(
    text: str,
    vote_count: Optional[int],
    base_url: Optional[str],
    model: Optional[str],
    temperature: Optional[float],
) -> Dict[str, Any]:
    """Fall back to one full semantic pass if segmentation cannot be trusted."""
    return parse_instruction_llm(
        text,
        fallback_to_rules=True,
        vote_count=1 if vote_count is None else vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def _valid_semantic_segments(
    text: str,
    segments: Optional[List[str]],
    max_sentences: int,
    max_phases: int = DEFAULT_SEGMENT_MAX_PHASES,
    max_words: int = DEFAULT_SEGMENT_MAX_WORDS,
) -> bool:
    """Require at least two ordered verbatim excerpts, each within all budgets."""
    if not segments or len(segments) < 2:
        return False
    if normalize_whitespace(" ".join(segments)) != normalize_whitespace(text):
        return False
    return all(
        bool(normalize_whitespace(segment))
        and not _exceeds_segment_budget(
            segment,
            max_sentences=max_sentences,
            max_phases=max_phases,
            max_words=max_words,
        )
        for segment in segments
    )


def _merged_result_requires_whole_parse(result: Dict[str, Any]) -> bool:
    """Reparse the full instruction when merged auditing finds a fidelity defect."""
    return (
        result.get("status") == "needs_review"
        and result.get("reason") in _FIDELITY_BLOCKING_REASONS
    )


def _merge_segment_results_into_plans(
    segment_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge local primary plans and lift retained step alternatives globally."""
    import json

    merged_tasks: List[Dict[str, Any]] = []
    merged_constraints: List[Dict[str, Any]] = []
    seen_constraints: set = set()
    lifted_candidates: List[tuple] = []
    offset = 0

    for result in segment_results:
        tasks = result.get("tasks", [])
        for task in tasks:
            merged_task = dict(task)
            merged_task["step_id"] = len(merged_tasks) + 1
            merged_tasks.append(merged_task)
        for constraint in result.get("constraints", []):
            key = json.dumps(constraint, sort_keys=True)
            if key not in seen_constraints:
                seen_constraints.add(key)
                merged_constraints.append(constraint)
        for group in result.get("backtracking", {}).get("step_candidates", []):
            global_sid = offset + group.get("step_id", 0)
            if 1 <= global_sid <= offset + len(tasks):
                for candidate in group.get("candidates", []):
                    lifted_candidates.append((global_sid, candidate))
        offset += len(tasks)

    if not merged_tasks:
        return []

    plans: List[Dict[str, Any]] = [{
        "candidate_id": "merged_long",
        "vote_support": 1,
        "tasks": merged_tasks,
        "constraints": merged_constraints,
    }]
    seen_signatures = {_merged_plan_signature(merged_tasks, merged_constraints)}

    for index, (step_id, candidate) in enumerate(lifted_candidates, start=1):
        candidate_tasks = [dict(task) for task in merged_tasks]
        replacement = dict(candidate)
        replacement.pop("rank", None)
        replacement["step_id"] = step_id
        candidate_tasks[step_id - 1] = replacement
        signature = _merged_plan_signature(candidate_tasks, merged_constraints)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        plans.append({
            "candidate_id": f"segment_alt_{index}",
            "vote_support": 1,
            "tasks": candidate_tasks,
            "constraints": list(merged_constraints),
        })

    return plans


def _merged_plan_signature(
    tasks: List[Dict[str, Any]],
    constraints: List[Dict[str, Any]],
) -> str:
    """Compare candidate meaning without treating confidence as semantics."""
    import json

    semantic_tasks = [
        {
            "action": task.get("action"),
            "direction": task.get("direction"),
            "features": task.get("features", []),
        }
        for task in tasks
    ]
    return json.dumps(
        {"tasks": semantic_tasks, "constraints": constraints},
        sort_keys=True,
    )


def _generate_initial_contribution_for_sentence(
    sentence: str,
    vote_count: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Generate initial tasks/constraints for a single sentence without global audit.

    Returns {"tasks": [...], "constraints": [...]} on success, None on failure.
    Falls back to rule parser if LLM produces no usable output.
    """
    from . import llm, aggregator

    # 1. LLM voting
    success, raw_votes = llm.parse_with_llm(
        instruction=sentence,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

    if success and raw_votes:
        plans = aggregator.collect_distinct_valid_plans(raw_votes)
        if plans:
            # Pick the plan with highest vote support
            best = max(plans, key=lambda p: p.get("vote_support", 0))
            return {
                "tasks": best.get("tasks", []),
                "constraints": best.get("constraints", []),
            }
        # A valid no-action sentence is a semantic no-op in long mode.
        from .semantic_compiler import compile_draft
        if any(
            compiled.get("status") == "ok" and not compiled.get("tasks")
            for compiled in (compile_draft(vote) for vote in raw_votes)
        ):
            return {"tasks": [], "constraints": []}

    # 2. Rule fallback for this single sentence
    rule_result = parse_instruction(sentence)
    if rule_result.get("tasks") or rule_result.get("constraints"):
        return {
            "tasks": rule_result.get("tasks", []),
            "constraints": rule_result.get("constraints", []),
        }

    return None


def _merge_initial_contributions(
    contributions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge per-sentence initial contributions into a single plan.

    Rules:
    - Tasks are concatenated in sentence order.
    - step_ids are renumbered continuously starting from 1.
    - Constraints are unioned and deduplicated.
    """
    import json

    merged_tasks: List[Dict[str, Any]] = []
    seen_constraints: set = set()
    merged_constraints: List[Dict[str, Any]] = []

    for contrib in contributions:
        tasks = contrib.get("tasks", [])
        for t in tasks:
            new_task = dict(t)
            new_task["step_id"] = len(merged_tasks) + 1
            merged_tasks.append(new_task)

        for c in contrib.get("constraints", []):
            key = json.dumps(c, sort_keys=True)
            if key not in seen_constraints:
                seen_constraints.add(key)
                merged_constraints.append(c)

    return {
        "candidate_id": "merged",
        "tasks": merged_tasks,
        "constraints": merged_constraints,
    }


def _run_global_postprocess(
    instruction: str,
    plans: List[Dict[str, Any]],
    vote_count: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    retry_audit: bool = False,
    preserve_tasks_on_audit_failure: bool = False,
) -> Dict[str, Any]:
    """Run fidelity audit, refinement, and status decision on initial plan(s).

    Args:
        instruction: The original full instruction for audit context.
        plans: One or more candidate plans (each with candidate_id, tasks, constraints).
        retry_audit: If True, retry the initial audit once on failure.
        preserve_tasks_on_audit_failure: If True, return merged tasks even when audit fails.
        **overrides: Backend overrides.

    Returns:
        Compact result dict.
    """
    from . import llm, aggregator

    if not plans:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={"step_candidates": []},
            reason="no_valid_plan",
        )

    # 1. Fidelity audit against original full instruction
    audits = llm.audit_plans_against_instruction(
        instruction, plans,
        base_url=base_url, model=model,
    )
    if audits is None and retry_audit:
        audits = llm.audit_plans_against_instruction(
            instruction, plans,
            base_url=base_url, model=model,
        )
    if audits is None:
        if preserve_tasks_on_audit_failure and plans:
            primary_plan = plans[0]
            return _make_compact_result(
                status="needs_review",
                confidence=0.0,
                tasks=primary_plan.get("tasks", []),
                constraints=primary_plan.get("constraints", []),
                backtracking={"step_candidates": []},
                reason="instruction_fidelity_audit_unavailable",
            )
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={"step_candidates": []},
            reason="instruction_fidelity_audit_unavailable",
        )

    # 2. Select primary plan
    primary = _select_primary_plan(plans, audits)
    if primary is None:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={"step_candidates": []},
            reason="no_valid_plan",
        )

    # 3. Write per-step confidences from audit
    primary = _write_audited_step_confidences(primary, audits)

    # 4. Detect global non-local competition
    global_risk = _detect_non_local_competition(primary, plans, audits)

    # 5. Refinement rounds for low-confidence steps
    replacement_plans: List[Dict[str, Any]] = []
    replacement_audits: List[Dict[str, Any]] = []

    for _round in range(MAX_PRIMARY_REFINEMENT_ROUNDS):
        target_steps = _find_low_confidence_steps(primary)
        if not target_steps:
            break

        generated = llm.generate_step_candidates(
            instruction,
            primary,
            target_steps,
            max_variants_per_step=MAX_GENERATED_VARIANTS_PER_STEP,
            base_url=base_url,
            model=model,
        )
        if generated is None:
            break

        optional_seeds = aggregator.extract_step_candidates(primary, plans)
        raw_candidates = _merge_and_deduplicate_candidates(generated, optional_seeds)
        if not raw_candidates:
            break

        replacement_plans = _build_replacement_plans(primary, raw_candidates)
        if not replacement_plans:
            break

        replacement_audits = llm.audit_plans_against_instruction(
            instruction, replacement_plans,
            base_url=base_url, model=model,
        )
        if replacement_audits is None:
            break

        better = _choose_better_primary_if_any(primary, replacement_plans, replacement_audits)
        if better is None:
            break

        primary = better

    # 6. Select backtracking candidates
    # If replacement_audits is None (audit failed during refinement), treat as empty.
    if replacement_audits is None:
        replacement_plans = []
        replacement_audits = []
    backtracking = _select_step_candidates(primary, replacement_plans, replacement_audits)

    # 7. Decide status
    status, reason = _decide_status(primary, backtracking, global_risk)

    # 8. Build result
    result = _make_compact_result(
        status=status,
        confidence=round(primary.get("plan_confidence", 0.0), 2),
        tasks=primary.get("tasks", []),
        constraints=primary.get("constraints", []),
        backtracking=backtracking,
        reason=reason,
    )

    try:
        validate_result(result)
    except ValueError:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={"step_candidates": []},
            reason="result_validation_failed",
        )

    return result


def _parse_single_sentence_llm(
    text: str,
    vote_count: Optional[int] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Parse a single sentence via fidelity-audit pipeline.

    Flow:
    1. LLM vote → collect distinct valid plans
    2. Hand off to _run_global_postprocess for audit, refinement, and status decision

    No pre-flight checks (caller already gated). No rule fallback.
    """
    from . import llm, aggregator

    # 1. LLM voting
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
            backtracking={"step_candidates": []},
            reason="initial_generation_failed",
        )

    # 2. Collect distinct valid plans with vote support
    plans = aggregator.collect_distinct_valid_plans(raw_votes)
    if not plans:
        return _make_compact_result(
            status="needs_review",
            confidence=0.0,
            tasks=[],
            constraints=[],
            backtracking={"step_candidates": []},
            reason="no_valid_candidate_plan",
        )

    # 3. Global postprocess: audit, refinement, status decision
    return _run_global_postprocess(
        text, plans,
        vote_count=vote_count,
        base_url=base_url,
        model=model,
        temperature=temperature,
        retry_audit=True,
        preserve_tasks_on_audit_failure=True,
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
        rule_result = parse_instruction(text)
        if rule_result.get("tasks") or rule_result.get("status") == "unsupported":
            return rule_result

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
        "backtracking": {"step_candidates": []},
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
    - 0.90 < C1 <= 0.95: save candidates where C1 - Cn <= 0.40
    - C1 <= 0.90: save candidates where C1 - Cn <= 0.50
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
            margin = STEP_MEDIUM_CANDIDATE_MARGIN
        else:
            margin = STEP_LOW_CANDIDATE_MARGIN

        kept: List[Dict[str, Any]] = []
        for rank, cand_task in enumerate(pool, start=2):
            if rank > 3:
                break
            conf = cand_conf.get((step_id, rank), 0.0)
            if _diff(c1, conf) <= margin:
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


# ── Fidelity-audit pipeline constants ───────────────────────────────────────
STEP_HIGH_CONFIDENCE = 0.95
STEP_LOW_CONFIDENCE = 0.90
STEP_MEDIUM_CANDIDATE_MARGIN = 0.40
STEP_LOW_CANDIDATE_MARGIN = 0.50
PLAN_MEDIUM_CANDIDATE_MARGIN = 0.15
PLAN_LOW_CANDIDATE_MARGIN = 0.20
MAX_SAVED_CANDIDATES_PER_STEP = 2
MAX_GENERATED_VARIANTS_PER_STEP = 3
MAX_PRIMARY_REFINEMENT_ROUNDS = 2
PLAN_ACCEPT_CONFIDENCE = 0.90
GLOBAL_AMBIGUITY_MARGIN = 0.15


def _select_primary_plan(
    plans: List[Dict[str, Any]],
    audits: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select the best primary plan from audited candidates.

    Priority:
    1. No blocking issues.
    2. Higher plan_confidence.
    3. Higher vote_support (tie-break).
    """
    audit_by_id = {a["candidate_id"]: a for a in audits}
    scored = []
    for p in plans:
        cid = p.get("candidate_id")
        audit = audit_by_id.get(cid)
        if audit is None:
            continue
        blocking = audit.get("blocking_issues", [])
        conf = audit.get("plan_confidence", 0.0)
        vote_support = p.get("vote_support", 0)
        scored.append((len(blocking) == 0, conf, vote_support, p, audit))

    if not scored:
        return None

    # Sort: no-blocking first, then confidence desc, then vote_support desc
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2]))
    _, _, _, best_plan, best_audit = scored[0]

    # Enrich plan with audit-derived plan_confidence
    primary = dict(best_plan)
    primary["plan_confidence"] = best_audit.get("plan_confidence", 0.0)
    primary["blocking_issues"] = list(best_audit.get("blocking_issues", []))
    return primary


def _write_audited_step_confidences(
    primary: Dict[str, Any],
    audits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Copy audit step confidences into primary tasks."""
    cid = primary.get("candidate_id")
    audit = next((a for a in audits if a.get("candidate_id") == cid), None)
    if audit is None:
        return primary

    step_conf_map = {
        sc["step_id"]: sc["confidence"]
        for sc in audit.get("step_confidences", [])
    }

    for t in primary.get("tasks", []):
        sid = t.get("step_id")
        if sid in step_conf_map:
            t["confidence"] = round(step_conf_map[sid], 2)

    return primary


def _detect_non_local_competition(
    primary: Dict[str, Any],
    plans: List[Dict[str, Any]],
    audits: List[Dict[str, Any]],
) -> bool:
    """Detect if there is a close competitor that cannot be expressed as a single-step diff."""
    from . import aggregator

    primary_conf = primary.get("plan_confidence", 0.0)
    audit_by_id = {a["candidate_id"]: a for a in audits}

    for p in plans:
        cid = p.get("candidate_id")
        if cid == primary.get("candidate_id"):
            continue
        audit = audit_by_id.get(cid)
        if audit is None:
            continue
        conf = audit.get("plan_confidence", 0.0)
        if primary_conf - conf < GLOBAL_AMBIGUITY_MARGIN:
            diff_type = aggregator.classify_plan_difference(primary, p)
            if diff_type != "single_task_diff":
                return True
    return False


def _find_low_confidence_steps(primary: Dict[str, Any]) -> List[int]:
    """Return step_ids with confidence <= STEP_HIGH_CONFIDENCE."""
    return [
        t["step_id"]
        for t in primary.get("tasks", [])
        if t.get("confidence", 1.0) <= STEP_HIGH_CONFIDENCE
    ]


def _merge_and_deduplicate_candidates(
    generated: Optional[Dict[int, List[Dict[str, Any]]]],
    optional_seeds: Optional[Dict[int, List[Dict[str, Any]]]],
) -> Dict[int, List[Dict[str, Any]]]:
    """Merge generated candidates with optional seeds, deduplicating by task signature."""
    import json

    def _task_sig(task: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "action": task.get("action"),
                "direction": task.get("direction"),
                "features": sorted(
                    [json.dumps(f, sort_keys=True) for f in task.get("features", [])]
                ),
            },
            sort_keys=True,
        )

    merged: Dict[int, List[Dict[str, Any]]] = {}

    for source in (generated, optional_seeds):
        if not source:
            continue
        for sid, cands in source.items():
            seen: set = set()
            out: List[Dict[str, Any]] = []
            for cand in merged.get(sid, []):
                sig = _task_sig(cand)
                if sig not in seen:
                    seen.add(sig)
                    out.append(cand)
            for cand in cands:
                sig = _task_sig(cand)
                if sig not in seen:
                    seen.add(sig)
                    out.append(cand)
            merged[sid] = out

    return merged


def _build_replacement_plans(
    primary: Dict[str, Any],
    raw_candidates: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Construct replacement plans by swapping each candidate into the primary plan."""
    replacement_plans: List[Dict[str, Any]] = []
    primary_tasks = primary.get("tasks", [])

    for sid, cands in raw_candidates.items():
        for cand in cands:
            new_tasks = []
            for t in primary_tasks:
                if t.get("step_id") == sid:
                    new_task = dict(t)
                    new_task["action"] = cand.get("action", t.get("action"))
                    if "direction" in cand:
                        new_task["direction"] = cand["direction"]
                    elif "direction" not in cand and t.get("direction") is not None:
                        # If candidate omits direction, keep primary direction unless action changed
                        pass
                    if "features" in cand:
                        new_task["features"] = list(cand["features"])
                    new_tasks.append(new_task)
                else:
                    new_tasks.append(dict(t))

            rep_plan: Dict[str, Any] = {
                "candidate_id": f"rep_{sid}_{len(replacement_plans)}",
                "tasks": new_tasks,
                "constraints": list(primary.get("constraints", [])),
            }
            replacement_plans.append(rep_plan)

    return replacement_plans


def _choose_better_primary_if_any(
    primary: Dict[str, Any],
    replacement_plans: List[Dict[str, Any]],
    replacement_audits: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a better primary plan if any replacement outperforms the current one."""
    primary_conf = primary.get("plan_confidence", 0.0)
    audit_by_id = {a["candidate_id"]: a for a in replacement_audits}

    best = None
    best_conf = primary_conf

    for rp in replacement_plans:
        cid = rp.get("candidate_id")
        audit = audit_by_id.get(cid)
        if audit is None:
            continue
        if audit.get("blocking_issues"):
            continue

        conf = audit.get("plan_confidence", 0.0)
        if conf > best_conf:
            best = (rp, audit)
            best_conf = conf

    if best is None:
        return None

    rp, audit = best
    new_primary = dict(rp)
    new_primary["plan_confidence"] = audit.get("plan_confidence", 0.0)
    new_primary["blocking_issues"] = list(audit.get("blocking_issues", []))
    new_primary = _write_audited_step_confidences(new_primary, [audit])
    return new_primary


def _select_step_candidates(
    primary: Dict[str, Any],
    replacement_plans: List[Dict[str, Any]],
    replacement_audits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Filter and rank step candidates using strict plan and relaxed step gates."""
    primary_conf = primary.get("plan_confidence", 0.0)
    audit_by_id = {a["candidate_id"]: a for a in replacement_audits}
    step_conf_map = {
        t["step_id"]: t.get("confidence", 1.0)
        for t in primary.get("tasks", [])
    }

    def _diff(a: float, b: float) -> float:
        return round(a - b, 4)

    step_candidates: List[Dict[str, Any]] = []

    # Group audits by target step_id (parse candidate_id like "rep_2_5")
    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for rp in replacement_plans:
        cid = rp.get("candidate_id", "")
        parts = cid.split("_")
        if len(parts) < 2:
            continue
        try:
            sid = int(parts[1])
        except ValueError:
            continue
        audit = audit_by_id.get(cid)
        if audit is None:
            continue
        by_step.setdefault(sid, []).append({"plan": rp, "audit": audit})

    for sid in sorted(by_step.keys()):
        s1 = step_conf_map.get(sid, 1.0)

        if s1 > STEP_HIGH_CONFIDENCE:
            continue

        if s1 > STEP_LOW_CONFIDENCE:
            step_margin = STEP_MEDIUM_CANDIDATE_MARGIN
            plan_margin = PLAN_MEDIUM_CANDIDATE_MARGIN
        else:
            step_margin = STEP_LOW_CANDIDATE_MARGIN
            plan_margin = PLAN_LOW_CANDIDATE_MARGIN

        kept: List[Dict[str, Any]] = []
        for item in by_step[sid]:
            audit = item["audit"]
            if audit.get("blocking_issues"):
                continue

            # Gate 1: replacement plan overall fidelity
            rep_conf = audit.get("plan_confidence", 0.0)
            if _diff(primary_conf, rep_conf) >= plan_margin:
                continue

            # Gate 2: candidate step confidence
            step_audit = next(
                (sc for sc in audit.get("step_confidences", []) if sc.get("step_id") == sid),
                None,
            )
            if step_audit is None:
                continue
            sn = step_audit.get("confidence", 0.0)
            if _diff(s1, sn) > step_margin:
                continue

            # Extract the differing task from the replacement plan
            rep_tasks = item["plan"].get("tasks", [])
            cand_task = next((t for t in rep_tasks if t.get("step_id") == sid), None)
            if cand_task is None:
                continue

            kept.append({
                "task": cand_task,
                "sn": sn,
                "pn": rep_conf,
            })

        if not kept:
            continue

        # Sort by Sn desc, then Pn desc
        kept.sort(key=lambda x: (-x["sn"], -x["pn"]))

        candidates_out: List[Dict[str, Any]] = []
        for rank, item in enumerate(kept[:MAX_SAVED_CANDIDATES_PER_STEP], start=2):
            ct = item["task"]
            candidates_out.append({
                "rank": rank,
                "step_id": sid,
                "action": ct.get("action", "UNKNOWN"),
                "direction": ct.get("direction"),
                "features": ct.get("features", []),
                "confidence": round(item["sn"], 2),
            })

        if candidates_out:
            step_candidates.append({
                "step_id": sid,
                "candidates": candidates_out,
            })

    return {"step_candidates": step_candidates}


def _decide_status(
    primary: Dict[str, Any],
    backtracking: Dict[str, Any],
    global_risk: bool,
) -> tuple:
    """Decide final status and reason."""
    plan_conf = primary.get("plan_confidence", 0.0)
    blocking = primary.get("blocking_issues", [])
    tasks = primary.get("tasks", [])

    if not tasks:
        return "none", "no_action"

    if blocking:
        return "needs_review", blocking[0]

    if global_risk:
        return "needs_review", "non_localizable_plan_competition"

    if plan_conf <= PLAN_ACCEPT_CONFIDENCE:
        return "needs_review", "plan_fidelity_below_threshold"

    # Check low-confidence steps without any backtracking candidates
    for t in tasks:
        conf = t.get("confidence", 1.0)
        if conf <= STEP_LOW_CONFIDENCE:
            sid = t.get("step_id")
            groups = backtracking.get("step_candidates", [])
            has_candidate = any(g.get("step_id") == sid for g in groups)
            if not has_candidate:
                return "needs_review", "low_confidence_step_without_candidate"

    return "ok", None


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
        merged["backtracking"] = {"step_candidates": []}

    return merged


def _make_compact_result(
    status: str,
    confidence: float,
    tasks: List[Dict[str, Any]],
    constraints: List[Dict[str, Any]],
    backtracking: Dict[str, Any],
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compact result dict.

    Canonical empty backtracking is always {"step_candidates": []}.
    """
    # Normalize empty backtracking to canonical form
    if not backtracking or "step_candidates" not in backtracking:
        backtracking = {"step_candidates": []}
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
