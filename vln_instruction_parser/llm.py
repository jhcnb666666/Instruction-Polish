"""LLM provider for VLN instruction parsing.

Supports three backends:
1. Local Qwen3-VL (default local backend, tested with Qwen3-VL-32B-Instruct)
2. Local InternVLA-N1 (legacy, not suitable for text generation)
3. Remote API (OpenAI-compatible, e.g. DashScope Qwen)

Backend is selected via environment variable VLN_LLM_BACKEND.
"""

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ── Backend selection ──────────────────────────────────────────────────────
DEFAULT_BACKEND = os.getenv("VLN_LLM_BACKEND", "local")

# ── Local model defaults ───────────────────────────────────────────────────
DEFAULT_LOCAL_MODEL_PATH = "/home/ubuntu/model/Qwen3-VL-32B-Instruct"
DEFAULT_LOCAL_VOTE_COUNT = 3
DEFAULT_LOCAL_TEMPERATURE = 0.3
DEFAULT_LOCAL_MAX_TOKENS = 1536

# ── Remote API defaults ────────────────────────────────────────────────────
DEFAULT_REMOTE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_REMOTE_MODEL = "qwen2.5-7b-instruct"
DEFAULT_REMOTE_VOTE_COUNT = 3
DEFAULT_REMOTE_TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a 2D VLN (Vision-Language Navigation) instruction parser.

Core Rules:
- Only English input is supported.
- Only 2D navigation is supported (walk, go, turn, stop, enter, exit, pass, face, wait).
- Supported directions: left, right, forward, backward, straight, around.
- Supported spatial relations: near, in_front_of, behind, left_of, right_of, inside, into, outside, through, along, toward, away_from, at, between, end_of, past.
- Do NOT parse 3D/vertical instructions (upstairs, downstairs, take the elevator, floor, climb, descend) as confident 2D actions. If they appear, set action=UNKNOWN and keep confidence low.
- For ambiguous instructions (e.g., "go over there"), set low confidence.

Execution-Order Rules (CRITICAL):
1. Read the ENTIRE instruction before creating any actions.
2. Resolve execution order from temporal expressions such as: before, after, once, until, then, finally.
3. Resolve replacement or negation expressions such as: instead of, rather than, do not, without.
4. Do NOT create actions for connectors such as: and, then, instead, before, after.
5. Output only actions that SHOULD actually be executed.
6. Assign id by EXECUTION ORDER, not by text appearance order.
7. If the sentence is grammatically malformed or ordering remains ambiguous, return the most plausible order with low confidence.

Feature Role Rules:
Valid roles: path, where, progress, target, terminate, start. ONLY use these six roles.
- "follow the hallway" / "go along the hallway" -> role="path", relation="along"
- "go to the door" -> role="target", relation="toward"
- "walk past the table" -> role="progress", relation="past"
- "turn left at the sofa" -> role="where", relation="at"
- "until you see/reach the sofa" -> role="terminate", trigger="see"/"reach"
- "when you see the sofa, turn left" -> role="start", trigger="see"
- "instead of entering the kitchen" -> exclude that action
- "do not enter the room" -> if safety-critical, add to constraints
- Descriptive side-keeping (e.g., "keeping X on your left") should be mapped to role="path", relation="left_of" or omitted if it does not define a navigational action

Output Format:
Return ONLY a JSON object with these keys:
- "actions": array of action objects
- "order": array of { "before": "id", "after": "id" }
- "constraints": array of { "type": "forbidden_action", "action": "...", "features": [...] }
- "excluded": array of excluded action ids or { "id": "...", "reason": "..." }

Each action object must have:
- id: unique string (e.g., "a1", "a2")
- action: string, one of [MOVE_FORWARD, TURN, GO_TO, PASS, ENTER, EXIT, STOP, FACE, WAIT, UNKNOWN]
- direction: string or omitted (one of [left, right, forward, backward, straight, around])
- features: array of { "role": "...", "relation": "...", "landmark": "...", "trigger": "..." }
- confidence: number between 0.0 and 1.0

Few-shot Examples:

Example 1:
Instruction: "Before turning left, go straight down the hallway."
{
  "actions": [
    {"id":"a1","action":"MOVE_FORWARD","direction":"straight","features":[{"role":"path","relation":"along","landmark":"hallway"}],"confidence":0.95},
    {"id":"a2","action":"TURN","direction":"left","features":[],"confidence":0.95}
  ],
  "order": [{"before":"a1","after":"a2"}],
  "constraints": [],
  "excluded": []
}

Example 2:
Instruction: "Turn left after passing the sofa."
{
  "actions": [
    {"id":"a1","action":"PASS","features":[{"role":"progress","relation":"past","landmark":"sofa"}],"confidence":0.93},
    {"id":"a2","action":"TURN","direction":"left","features":[],"confidence":0.93}
  ],
  "order": [{"before":"a1","after":"a2"}],
  "constraints": [],
  "excluded": []
}

Example 3:
Instruction: "Instead of entering the kitchen, turn right at the door."
{
  "actions": [
    {"id":"a1","action":"TURN","direction":"right","features":[{"role":"where","relation":"at","landmark":"door"}],"confidence":0.94}
  ],
  "order": [],
  "constraints": [],
  "excluded": [
    {"id":"ex1","reason":"replaced_by_instead"}
  ]
}

Example 4:
Instruction: "Follow the hallway until you see the sofa on your left, then turn left at the sofa."
{
  "actions": [
    {"id":"a1","action":"MOVE_FORWARD","direction":"straight","features":[{"role":"path","relation":"along","landmark":"hallway"},{"role":"terminate","trigger":"see","relation":"left_of_agent","landmark":"sofa"}],"confidence":0.93},
    {"id":"a2","action":"TURN","direction":"left","features":[{"role":"where","relation":"at","landmark":"sofa"}],"confidence":0.93}
  ],
  "order": [{"before":"a1","after":"a2"}],
  "constraints": [],
  "excluded": []
}

Example 5:
Instruction: "Do not enter the room; wait outside the door."
{
  "actions": [
    {"id":"a1","action":"WAIT","features":[{"role":"where","relation":"outside","landmark":"door"}],"confidence":0.88}
  ],
  "order": [],
  "constraints": [
    {"type":"forbidden_action","action":"ENTER","features":[{"role":"where","relation":"inside","landmark":"room"}]}
  ],
  "excluded": []
}

Example 6:
Instruction: "Go down the stairs and stop in the middle of the landing."
{
  "actions": [
    {"id":"a1","action":"UNKNOWN","features":[],"confidence":0.1}
  ],
  "order": [],
  "constraints": [],
  "excluded": []
}

Output ONLY the JSON object. Do NOT wrap it in markdown code blocks.
"""

ADJUDICATION_SYSTEM_PROMPT = """You are a VLN instruction adjudicator.

Your job is to decide the correct execution plan of 2D navigation tasks given conflicting interpretations of the same instruction.

Rules:
- Only 2D navigation actions: MOVE_FORWARD, TURN, GO_TO, PASS, ENTER, EXIT, STOP, FACE, WAIT, UNKNOWN.
- Temporal words (before, after, until, once) determine execution order, not text order.
- "Instead of X, do Y" means X is excluded; only Y is executed.
- "Do not X" means X is excluded.
- 3D/vertical actions (stairs, elevator, floor) should be marked unsupported.
- Return ONLY a JSON object with a single key "execution_plan" containing an array of step objects in execution order.

Each step object must have:
- action: string
- direction: string or omitted
- features: array of { "role": "...", "relation": "...", "landmark": "...", "trigger": "..." }

Example:
Instruction: "Before turning left, go straight."
Candidate A plan: [{"action":"TURN","direction":"left","features":[]}, {"action":"MOVE_FORWARD","direction":"straight","features":[]}]
Candidate B plan: [{"action":"MOVE_FORWARD","direction":"straight","features":[]}, {"action":"TURN","direction":"left","features":[]}]
Your output: {"execution_plan": [{"action":"MOVE_FORWARD","direction":"straight","features":[]}, {"action":"TURN","direction":"left","features":[]}]}

Output ONLY the JSON object.
"""

VERIFIER_SYSTEM_PROMPT = """You are a VLN plan verifier.

Your job is to rank candidate navigation plans for a given instruction by confidence.

Rules:
- Only 2D navigation actions: MOVE_FORWARD, TURN, GO_TO, PASS, ENTER, EXIT, STOP, FACE, WAIT, UNKNOWN.
- Review the original instruction and each candidate plan carefully.
- Assign confidence scores between 0.0 and 1.0 based on how well each candidate matches the instruction.
- You may ONLY reorder and score existing candidates. Do NOT create, modify, or add tasks, features, or constraints.
- Return ONLY a JSON object with a single key "ranked_candidates".

Output format:
{
  "ranked_candidates": [
    {"candidate_id": "p1", "confidence": 0.94},
    {"candidate_id": "p2", "confidence": 0.81}
  ]
}

Output ONLY the JSON object.
"""

STEP_VERIFIER_SYSTEM_PROMPT = """You are a step-level confidence verifier for 2D VLN navigation instructions.

Given the original instruction, the primary execution plan, and optional alternative step candidates,
rate the semantic confidence of each primary step and each candidate step.

Rules:
- Confidence must be between 0.0 and 1.0.
- Higher confidence means the step more accurately reflects the instruction semantics.
- Review the original instruction carefully before rating.
- Rate primary steps independently based on how well they match the instruction.
- Rate candidate steps relative to their primary counterpart; if a candidate changes the meaning significantly, assign low confidence.
- Return ONLY a JSON object with "step_confidences" and "candidate_confidences".

Output format:
{
  "step_confidences": [
    {"step_id": 1, "confidence": 0.97},
    {"step_id": 2, "confidence": 0.93}
  ],
  "candidate_confidences": [
    {"step_id": 2, "rank": 2, "confidence": 0.84}
  ]
}

Output ONLY the JSON object.
"""


def _get_config() -> Dict[str, Any]:
    backend = os.getenv("VLN_LLM_BACKEND", DEFAULT_BACKEND).lower()
    if backend == "local":
        return {
            "backend": "local",
            "model_path": os.getenv("VLN_LLM_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH),
            "vote_count": int(os.getenv("VLN_LLM_VOTE_COUNT", str(DEFAULT_LOCAL_VOTE_COUNT))),
            "temperature": float(os.getenv("VLN_LLM_TEMPERATURE", str(DEFAULT_LOCAL_TEMPERATURE))),
            "max_tokens": int(os.getenv("VLN_LLM_MAX_TOKENS", str(DEFAULT_LOCAL_MAX_TOKENS))),
        }
    else:
        return {
            "backend": "remote",
            "base_url": os.getenv("VLN_LLM_BASE_URL", DEFAULT_REMOTE_BASE_URL).rstrip("/"),
            "api_key": os.getenv("VLN_LLM_API_KEY", ""),
            "model": os.getenv("VLN_LLM_MODEL", DEFAULT_REMOTE_MODEL),
            "vote_count": int(os.getenv("VLN_LLM_VOTE_COUNT", str(DEFAULT_REMOTE_VOTE_COUNT))),
            "temperature": float(os.getenv("VLN_LLM_TEMPERATURE", str(DEFAULT_REMOTE_TEMPERATURE))),
        }


def _build_prompt(instruction: str) -> str:
    return (
        f'Parse the following English navigation instruction into ordered 2D navigation actions.\n\n'
        f'Instruction: "{instruction}"\n\n'
        f'Return ONLY a JSON object with "actions", "order", "constraints", and "excluded" arrays.\n\nJSON:'
    )


def _build_adjudication_prompt(instruction: str, candidate_plans: List[List[Dict[str, Any]]]) -> str:
    lines = [
        f'Instruction: "{instruction}"',
        '',
        'Conflicting candidate execution plans:',
    ]
    for i, plan in enumerate(candidate_plans, start=1):
        lines.append(f'Candidate {i}: {json.dumps(plan)}')
    lines.extend([
        '',
        'Based on the instruction\'s temporal logic (before/after/instead of/do not), decide the correct execution plan.',
        'Return ONLY a JSON object: {"execution_plan": [{"action":"...","direction":"...","features":[...]}, ...]}',
    ])
    return '\n'.join(lines)


def _build_verifier_prompt(instruction: str, candidates: List[Dict[str, Any]]) -> str:
    lines = [
        f'Instruction: "{instruction}"',
        '',
        'Candidate plans:',
    ]
    for c in candidates:
        cid = c.get("candidate_id", "unknown")
        lines.append(f'  {cid}: {json.dumps(c, default=str)}')
    lines.extend([
        '',
        'Rank these candidates by how well they match the instruction.',
        'Return ONLY a JSON object: {"ranked_candidates": [{"candidate_id":"p1","confidence":0.94}, ...]}',
    ])
    return '\n'.join(lines)


def _build_step_verifier_prompt(
    instruction: str,
    primary_tasks: List[Dict[str, Any]],
    step_candidate_pools: Dict[int, List[Dict[str, Any]]],
) -> str:
    lines = [
        f'Instruction: "{instruction}"',
        '',
        'Primary execution plan:',
    ]
    for t in primary_tasks:
        lines.append(f'  step {t["step_id"]}: {json.dumps(t, default=str)}')

    if step_candidate_pools:
        lines.extend(['', 'Alternative step candidates:'])
        for step_id, cands in sorted(step_candidate_pools.items()):
            for i, cand in enumerate(cands, start=2):
                lines.append(f'  step {step_id} (alternative {i}): {json.dumps(cand, default=str)}')

    lines.extend([
        '',
        'Rate the confidence of each primary step and each alternative step.',
        'Return ONLY a JSON object: {"step_confidences": [{"step_id":1,"confidence":0.97},...], "candidate_confidences": [{"step_id":2,"rank":2,"confidence":0.84},...]}',
    ])
    return '\n'.join(lines)


def verify_candidate_plans(
    instruction: str,
    candidates: List[Dict[str, Any]],
    **overrides: Any,
) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to rank and score up to three candidate compact plans.

    Args:
        instruction: Original instruction.
        candidates: List of compact candidate plan dicts, each with "candidate_id".
        **overrides: Backend overrides.

    Returns:
        List of {"candidate_id": str, "confidence": float} sorted by confidence
        descending, or None if verifier fails.
    """
    if not candidates:
        return None

    cfg = _get_config()
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    prompt = _build_verifier_prompt(instruction, candidates)
    system_prompt = VERIFIER_SYSTEM_PROMPT

    if cfg["backend"] == "local":
        result = _call_local(
            instruction,
            cfg["model_path"],
            temperature=0.0,
            max_tokens=cfg["max_tokens"],
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
    else:
        url = f"{cfg['base_url']}/chat/completions"
        payload = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                body = resp.read().decode("utf-8")
                response_json = json.loads(body)
            content = response_json["choices"][0]["message"]["content"]
            result = json.loads(content)
        except Exception:
            return None

    if result is None:
        return None

    ranked = result.get("ranked_candidates")
    if not isinstance(ranked, list):
        return None

    # Validate: candidate_id must exist in input, confidence in [0,1], no duplicates
    valid_ids = {c.get("candidate_id") for c in candidates}
    seen_ids: set = set()
    out: List[Dict[str, Any]] = []
    for item in ranked:
        if not isinstance(item, dict):
            continue
        cid = item.get("candidate_id")
        conf = item.get("confidence")
        if cid not in valid_ids:
            return None
        if cid in seen_ids:
            return None
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            return None
        seen_ids.add(cid)
        out.append({"candidate_id": cid, "confidence": float(conf)})

    if not out:
        return None

    # Must contain exactly all input candidate_ids
    if seen_ids != valid_ids:
        return None

    # Sort by confidence descending (stable)
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out


def verify_step_candidates(
    instruction: str,
    primary_tasks: List[Dict[str, Any]],
    step_candidate_pools: Dict[int, List[Dict[str, Any]]],
    **overrides: Any,
) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to rate confidence of each primary step and each step candidate.

    Args:
        instruction: Original instruction.
        primary_tasks: List of primary task dicts (the chosen main plan).
        step_candidate_pools: Dict mapping step_id -> list of candidate task dicts.
        **overrides: Backend overrides.

    Returns:
        Dict with "step_confidences" and "candidate_confidences", or None if verifier fails.
    """
    if not primary_tasks:
        return None

    cfg = _get_config()
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    prompt = _build_step_verifier_prompt(instruction, primary_tasks, step_candidate_pools)
    system_prompt = STEP_VERIFIER_SYSTEM_PROMPT

    if cfg["backend"] == "local":
        result = _call_local(
            instruction,
            cfg["model_path"],
            temperature=0.0,
            max_tokens=cfg["max_tokens"],
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
    else:
        url = f"{cfg['base_url']}/chat/completions"
        payload = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                body = resp.read().decode("utf-8")
                response_json = json.loads(body)
            content = response_json["choices"][0]["message"]["content"]
            result = json.loads(content)
        except Exception:
            return None

    if result is None:
        return None

    step_confidences = result.get("step_confidences")
    candidate_confidences = result.get("candidate_confidences")
    if not isinstance(step_confidences, list) or not isinstance(candidate_confidences, list):
        return None

    # Validate step_confidences
    primary_step_ids = {t.get("step_id") for t in primary_tasks}
    seen_step_ids: set = set()
    out_steps: List[Dict[str, Any]] = []
    for item in step_confidences:
        if not isinstance(item, dict):
            continue
        sid = item.get("step_id")
        conf = item.get("confidence")
        if sid not in primary_step_ids:
            return None
        if sid in seen_step_ids:
            return None
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            return None
        seen_step_ids.add(sid)
        out_steps.append({"step_id": sid, "confidence": float(conf)})

    # Validate candidate_confidences
    out_cands: List[Dict[str, Any]] = []
    for item in candidate_confidences:
        if not isinstance(item, dict):
            continue
        sid = item.get("step_id")
        rank = item.get("rank")
        conf = item.get("confidence")
        if sid not in primary_step_ids:
            return None
        if not isinstance(rank, int) or rank not in (2, 3):
            return None
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            return None
        out_cands.append({"step_id": sid, "rank": rank, "confidence": float(conf)})

    return {
        "step_confidences": out_steps,
        "candidate_confidences": out_cands,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  LOCAL BACKEND
# ═════════════════════════════════════════════════════════════════════════════

_local_model = None
_local_processor = None
_local_model_type = None


def _detect_model_type(model_path: str) -> str:
    """Detect model type from config.json."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("model_type", "").lower()
    return ""


def _load_local_model(model_path: str):
    """Lazy-load the local model and processor/tokenizer."""
    global _local_model, _local_processor, _local_model_type
    if _local_model is not None and _local_processor is not None:
        return _local_model, _local_processor, _local_model_type

    import torch

    model_type = _detect_model_type(model_path)
    _local_model_type = model_type

    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    elif model_type == "internvla_n1":
        import sys
        sys.path.insert(0, "/home/ubuntu/project/InternNav-lora")

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ForCausalLM,
            InternVLAN1ModelConfig,
        )

        AutoConfig.register("internvla_n1", InternVLAN1ModelConfig)
        AutoModelForCausalLM.register(InternVLAN1ModelConfig, InternVLAN1ForCausalLM)

        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    else:
        # Generic fallback: try AutoModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    _local_model = model
    _local_processor = processor
    return model, processor, model_type


def _generate_qwen3vl(
    model,
    processor,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """Generate with Qwen3-VL using chat template."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return response.strip()


def _generate_generic(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """Generate with generic CausalLM."""
    import torch

    prompt = f"{system_prompt}\n\n{user_prompt}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def _call_local(
    instruction: str,
    model_path: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    try:
        model, processor, model_type = _load_local_model(model_path)
    except Exception:
        return None

    if user_prompt is None:
        user_prompt = _build_prompt(instruction)

    if model_type == "qwen3_vl":
        response = _generate_qwen3vl(
            model, processor, system_prompt, user_prompt, temperature, max_tokens
        )
    else:
        response = _generate_generic(
            model, processor, system_prompt, user_prompt, temperature, max_tokens
        )

    if response is None:
        return None

    parsed = _extract_json(response)
    return parsed


# ═════════════════════════════════════════════════════════════════════════════
#  REMOTE BACKEND
# ═════════════════════════════════════════════════════════════════════════════

def _call_remote(
    instruction: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float = 120.0,
    system_prompt: str = SYSTEM_PROMPT,
    user_content: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None

    url = f"{base_url}/chat/completions"
    if user_content is None:
        user_content = _build_prompt(instruction)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            response_json = json.loads(body)
    except Exception:
        return None

    try:
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError):
        return None

    if not isinstance(parsed, dict) or "actions" not in parsed:
        return None

    return parsed


# ═════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from arbitrary text."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*?\"actions\"[\s\S]*?\}", text)
    if match:
        start = match.start()
        brace_count = 0
        end = start
        for i, ch in enumerate(text[start:]):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


def _call_backend(
    instruction: str,
    cfg: Dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
) -> Optional[Dict[str, Any]]:
    """Route to the appropriate backend."""
    if cfg["backend"] == "local":
        return _call_local(
            instruction,
            cfg["model_path"],
            cfg["temperature"],
            cfg["max_tokens"],
            system_prompt=system_prompt,
        )
    else:
        return _call_remote(
            instruction,
            cfg["base_url"],
            cfg["api_key"],
            cfg["model"],
            cfg["temperature"],
            system_prompt=system_prompt,
        )


def parse_with_llm(
    instruction: str,
    vote_count: Optional[int] = None,
    **overrides: Any,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Parse an instruction using the configured LLM backend with voting.

    Returns:
        (success, list_of_raw_votes): Each raw vote is a dict with
        "actions", "order", "constraints", "excluded".
    """
    cfg = _get_config()
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    vc = vote_count if vote_count is not None else cfg["vote_count"]

    raw_votes: List[Dict[str, Any]] = []
    for _ in range(vc):
        result = _call_backend(instruction, cfg)
        if result is None:
            return False, []
        actions = result.get("actions", [])
        if not isinstance(actions, list):
            return False, []
        for i, a in enumerate(actions):
            if "id" not in a:
                a["id"] = f"a{i+1}"
        raw_votes.append(result)

    return True, raw_votes


def adjudicate_plan(
    instruction: str,
    candidate_plans: List[List[Dict[str, Any]]],
    **overrides: Any,
) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to choose between conflicting execution plans.

    Args:
        instruction: Original instruction.
        candidate_plans: List of candidate plans, each a list of step dicts
                         with keys action, direction, features.
        **overrides: Backend overrides.

    Returns:
        The chosen execution plan as a list of step dicts, or None.
    """
    cfg = _get_config()
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    prompt = _build_adjudication_prompt(instruction, candidate_plans)
    system_prompt = ADJUDICATION_SYSTEM_PROMPT

    if cfg["backend"] == "local":
        result = _call_local(
            instruction,
            cfg["model_path"],
            temperature=0.0,
            max_tokens=cfg["max_tokens"],
            system_prompt=system_prompt,
            user_prompt=prompt,
        )
    else:
        url = f"{cfg['base_url']}/chat/completions"
        payload = {
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                body = resp.read().decode("utf-8")
                response_json = json.loads(body)
            content = response_json["choices"][0]["message"]["content"]
            result = json.loads(content)
        except Exception:
            return None

    if result is None:
        return None

    plan = result.get("execution_plan")
    if isinstance(plan, list) and all(isinstance(step, dict) for step in plan):
        return plan
    return None
