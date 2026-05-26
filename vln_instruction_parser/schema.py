"""Compact schema definitions for VLN instruction parser output."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Feature:
    role: str
    relation: Optional[str] = None
    landmark: Optional[str] = None
    trigger: Optional[str] = None


@dataclass
class Task:
    step_id: int
    action: str
    features: List[Feature] = field(default_factory=list)
    direction: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Constraint:
    type: str
    action: str
    features: List[Feature] = field(default_factory=list)
    direction: Optional[str] = None


@dataclass
class AlternativePlan:
    rank: int
    confidence: float
    tasks: List[Task] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)


@dataclass
class BacktrackingCandidate:
    rank: int
    step_id: int
    action: str
    features: List[Feature] = field(default_factory=list)
    direction: Optional[str] = None
    confidence: float = 1.0


@dataclass
class StepCandidateGroup:
    step_id: int
    candidates: List[BacktrackingCandidate] = field(default_factory=list)


@dataclass
class BacktrackingResult:
    step_candidates: List[StepCandidateGroup] = field(default_factory=list)


@dataclass
class ParseResult:
    status: str
    confidence: float
    tasks: List[Task] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    alternatives: List[AlternativePlan] = field(default_factory=list)
    backtracking: BacktrackingResult = field(default_factory=BacktrackingResult)
    reason: Optional[str] = None


def _feature_to_dict(feature: Feature) -> dict:
    d: Dict[str, Optional[str]] = {"role": feature.role}
    if feature.relation is not None:
        d["relation"] = feature.relation
    if feature.landmark is not None:
        d["landmark"] = feature.landmark
    if feature.trigger is not None:
        d["trigger"] = feature.trigger
    return d


def _feature_from_dict(d: dict) -> Feature:
    return Feature(
        role=d["role"],
        relation=d.get("relation"),
        landmark=d.get("landmark"),
        trigger=d.get("trigger"),
    )


def task_to_dict(task: Task) -> dict:
    d: Dict[str, object] = {
        "step_id": task.step_id,
        "action": task.action,
        "features": [_feature_to_dict(f) for f in task.features],
        "confidence": round(task.confidence, 2),
    }
    if task.direction is not None:
        d["direction"] = task.direction
    return d


def task_from_dict(d: dict) -> Task:
    return Task(
        step_id=d["step_id"],
        action=d["action"],
        direction=d.get("direction"),
        confidence=d.get("confidence", 1.0),
        features=[_feature_from_dict(f) for f in d.get("features", [])],
    )


def constraint_to_dict(c: Constraint) -> dict:
    d: Dict[str, object] = {
        "type": c.type,
        "action": c.action,
        "features": [_feature_to_dict(f) for f in c.features],
    }
    if c.direction is not None:
        d["direction"] = c.direction
    return d


def constraint_from_dict(d: dict) -> Constraint:
    return Constraint(
        type=d["type"],
        action=d["action"],
        direction=d.get("direction"),
        features=[_feature_from_dict(f) for f in d.get("features", [])],
    )


def alternative_plan_to_dict(alt: AlternativePlan) -> dict:
    return {
        "rank": alt.rank,
        "confidence": round(alt.confidence, 2),
        "tasks": [task_to_dict(t) for t in alt.tasks],
        "constraints": [constraint_to_dict(c) for c in alt.constraints],
    }


def alternative_plan_from_dict(d: dict) -> AlternativePlan:
    return AlternativePlan(
        rank=d["rank"],
        confidence=d["confidence"],
        tasks=[task_from_dict(t) for t in d.get("tasks", [])],
        constraints=[constraint_from_dict(c) for c in d.get("constraints", [])],
    )


def backtracking_candidate_to_dict(bc: BacktrackingCandidate) -> dict:
    d: Dict[str, object] = {
        "rank": bc.rank,
        "step_id": bc.step_id,
        "action": bc.action,
        "features": [_feature_to_dict(f) for f in bc.features],
        "confidence": round(bc.confidence, 2),
    }
    if bc.direction is not None:
        d["direction"] = bc.direction
    return d


def backtracking_candidate_from_dict(d: dict) -> BacktrackingCandidate:
    return BacktrackingCandidate(
        rank=d["rank"],
        step_id=d["step_id"],
        action=d["action"],
        direction=d.get("direction"),
        confidence=d.get("confidence", 1.0),
        features=[_feature_from_dict(f) for f in d.get("features", [])],
    )


def step_candidate_group_to_dict(g: StepCandidateGroup) -> dict:
    return {
        "step_id": g.step_id,
        "candidates": [backtracking_candidate_to_dict(c) for c in g.candidates],
    }


def step_candidate_group_from_dict(d: dict) -> StepCandidateGroup:
    return StepCandidateGroup(
        step_id=d["step_id"],
        candidates=[backtracking_candidate_from_dict(c) for c in d.get("candidates", [])],
    )


def backtracking_result_to_dict(bt: BacktrackingResult) -> dict:
    return {
        "step_candidates": [step_candidate_group_to_dict(g) for g in bt.step_candidates],
    }


def backtracking_result_from_dict(d: dict) -> BacktrackingResult:
    return BacktrackingResult(
        step_candidates=[step_candidate_group_from_dict(g) for g in d.get("step_candidates", [])],
    )


def result_to_dict(result: ParseResult) -> dict:
    d: Dict[str, object] = {
        "status": result.status,
        "confidence": round(result.confidence, 2),
        "tasks": [task_to_dict(t) for t in result.tasks],
        "constraints": [constraint_to_dict(c) for c in result.constraints],
        "alternatives": [alternative_plan_to_dict(a) for a in result.alternatives],
        "backtracking": backtracking_result_to_dict(result.backtracking),
    }
    if result.reason is not None:
        d["reason"] = result.reason
    return d


def result_from_dict(d: dict) -> ParseResult:
    return ParseResult(
        status=d["status"],
        confidence=d["confidence"],
        tasks=[task_from_dict(t) for t in d.get("tasks", [])],
        constraints=[constraint_from_dict(c) for c in d.get("constraints", [])],
        alternatives=[alternative_plan_from_dict(a) for a in d.get("alternatives", [])],
        backtracking=backtracking_result_from_dict(d.get("backtracking", {})),
        reason=d.get("reason"),
    )
