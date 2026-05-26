"""Demo of LLM-based VLN instruction parser via Qwen API.

Prerequisites:
    1. Obtain a DashScope (Alibaba Cloud) API key:
       https://dashscope.console.aliyun.com/apiKey
    2. Set the environment variable:
       export VLN_LLM_API_KEY=sk-xxxxxxxx

Optional environment overrides:
    VLN_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    VLN_LLM_MODEL=qwen2.5-7b-instruct
    VLN_LLM_VOTE_COUNT=3
    VLN_LLM_TEMPERATURE=0.2

If the API is unavailable or the key is missing, the parser automatically
falls back to the rule-based engine for simple instructions, and returns
status="needs_review" for complex instructions.
"""

import json
import os
import sys

# Allow importing the local package without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vln_instruction_parser import parse_instruction_auto

SAMPLES = [
    "Go down the hallway, turn left in front of the sofa, then stop by the kitchen door.",
    "Before turning left, go straight down the hallway.",
    "Instead of entering the kitchen, go straight and stop beside the couch.",
    "Go down the stairs and stop in the middle of the landing.",
]


def main():
    print("=" * 70)
    print("VLN Instruction Parser - Semantic Parsing Demo")
    print("=" * 70)

    api_key = os.getenv("VLN_LLM_API_KEY", "")
    if not api_key:
        print("\nWARNING: VLN_LLM_API_KEY not set. LLM will be unavailable.")
        print("         Simple instructions will use rule-based fallback.")
        print("         Complex instructions will return status=needs_review.\n")

    for text in SAMPLES:
        print(f"\nInput:\n  {text}\n")
        try:
            result = parse_instruction_auto(text)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"Status: {result['status']}  |  Confidence: {result['confidence']}")
        if result.get("reason"):
            print(f"Reason: {result['reason']}")

        for t in result.get("tasks", []):
            feats = json.dumps(t.get("features", []), ensure_ascii=False)
            dir_part = f"  dir={t['direction']}" if t.get("direction") else ""
            print(
                f"  Step {t['step_id']}: {t['action']:15}{dir_part}  |  features={feats}"
            )

        if result.get("constraints"):
            print(f"  constraints: {json.dumps(result['constraints'], ensure_ascii=False)}")

        if result.get("alternatives"):
            print(f"  alternatives: {len(result['alternatives'])} plan(s)")

        bt = result.get("backtracking", {})
        step_candidates = bt.get("step_candidates", [])
        if step_candidates:
            print(f"  backtracking: {len(step_candidates)} step group(s)")
            for group in step_candidates:
                sid = group["step_id"]
                for cand in group.get("candidates", []):
                    dir_part = f" dir={cand['direction']}" if cand.get("direction") else ""
                    print(
                        f"    -> Step {sid} rank={cand['rank']}: {cand['action']}{dir_part}"
                    )

    print("\n" + "=" * 70)
    print("Full JSON output for last sample:")
    print("=" * 70)
    if 'result' in dir():
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
