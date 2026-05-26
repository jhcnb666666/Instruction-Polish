# Instruction Polish

A standalone project for polishing and optimizing AI instructions / prompts.

## Features

- **Multiple polishing strategies**: clarity, conciseness, detail, roleplay, default
- **Heuristic quality scoring**: rule-based score (0-100) with detailed reasons
- **Batch processing**: process a whole file of instructions at once
- **Custom strategies**: register your own strategy functions
- **Configurable**: JSON config file support
- **CLI and Python API**: use from command line or import in code

## Installation

```bash
pip install -e .
```

## Quick Start

### Python API

```python
from instruction_polish import polish, analyze, list_strategies, register_strategy

# Polish a single instruction
print(polish("can u help me write python code", strategy="clarity"))
# -> "Can you help me write python code."

# Chain multiple strategies
print(polish("u summarize this", strategy="clarity,detail"))
# -> "Please you summarize this, and explain your reasoning in detail."

# Analyze quality metrics and heuristic score
print(analyze("Please explain quantum computing step by step with examples."))
# -> {'length': 60, 'word_count': 9, 'score': 85.0, 'score_reasons': [...]}

# List available strategies
print(list_strategies())
# -> ['default', 'clarity', 'conciseness', 'detail', 'roleplay']

# Register a custom strategy
register_strategy("shout", lambda s: s.upper() + "!!!")
print(polish("hello world", strategy="shout"))
# -> "HELLO WORLD!!!"
```

### Command Line

```bash
# Polish a single instruction
instruction-polish "explain quantum computing" --strategy clarity

# Chain strategies
instruction-polish "u summarize this" --strategy clarity,detail

# Analyze an instruction (with heuristic score)
instruction-polish "explain quantum computing" --analyze

# Compare all strategies side-by-side
instruction-polish "explain quantum computing" --compare

# Show diff of changes
instruction-polish "can u help me write python code" --strategy clarity --diff

# Suggest the best strategy
instruction-polish "can u help me write python code" --suggest

# Automatically apply the best suggested strategy
instruction-polish "can u help me write python code" --best

# List strategies
instruction-polish --list-strategies

# Batch process a plain text file (one instruction per line)
instruction-polish --input-file examples/sample_instructions.txt \
                   --output-file out.json \
                   --strategy clarity

# Batch process a JSON Lines file (.jsonl)
instruction-polish --input-file examples/sample_instructions.jsonl \
                   --output-file out.jsonl \
                   --strategy clarity
```

## Strategies

| Strategy | Description |
|----------|-------------|
| `default` | Capitalize first letter and ensure punctuation |
| `clarity` | Remove filler words, fix informal abbreviations (u -> you, wanna -> want to, ...) |
| `conciseness` | Replace wordy phrases with concise alternatives |
| `detail` | Add politeness and request detailed explanation |
| `roleplay` | Wrap in an expert role-play context |
| `auto` | Automatically pick the best built-in strategy based on heuristic score |

## Project Structure

```
instruction_polish/
├── src/instruction_polish/   # Main package
│   ├── __init__.py
│   ├── __main__.py           # python -m instruction_polish
│   ├── core.py               # Core polish / analyze logic
│   ├── strategies.py         # Built-in strategies + registry
│   ├── scoring.py            # Heuristic quality scoring
│   ├── batch.py              # Batch processing
│   ├── config.py             # Config loader
│   ├── cli.py                # Command-line interface
│   └── utils.py              # Logging helpers
├── tests/                    # Unit tests
├── examples/                 # Sample data and configs
├── README.md
├── pyproject.toml
├── Makefile
└── requirements.txt
```

## Development

```bash
# Install in editable mode
make install

# Run tests
make test

# Run VLN parser tests
make test-vln

# Run example commands
make example

# Clean build artifacts
make clean
```

## VLN Instruction Parser

A semantic 2D Vision-Language Navigation instruction parser is also included in this repository under `vln_instruction_parser/`.

### Entry Points

```python
from vln_instruction_parser import parse_instruction_auto

# Recommended: automatically chooses rule or LLM parser
result = parse_instruction_auto(
    "Follow the hallway until you see the sofa, then turn left at the door."
)
```

Three parsers are available:
- `parse_instruction_auto(text)` — **Recommended**. Routes simple instructions to the rule engine and complex instructions to the LLM.
- `parse_instruction(text)` — Rule-based parser for simple, unambiguous instructions.
- `parse_instruction_llm(text)` — LLM-based parser with voting and semantic understanding.

### Compact Output Schema

```json
{
  "status": "ok | needs_review | unsupported | none",
  "confidence": 1.0,
  "tasks": [
    {
      "step_id": 1,
      "action": "MOVE_FORWARD",
      "direction": "straight",
      "features": [
        {
          "role": "path",
          "relation": "along",
          "landmark": "hallway"
        },
        {
          "role": "terminate",
          "trigger": "see",
          "relation": "left_of_agent",
          "landmark": "sofa"
        }
      ]
    },
    {
      "step_id": 2,
      "action": "TURN",
      "direction": "left",
      "features": [
        {
          "role": "where",
          "relation": "at",
          "landmark": "sofa"
        }
      ]
    }
  ],
  "constraints": [],
  "alternatives": [],
  "backtracking": {
    "step_candidates": [
      {
        "step_id": 2,
        "candidates": [
          {
            "rank": 2,
            "step_id": 2,
            "action": "TURN",
            "direction": "right",
            "features": [],
            "confidence": 0.82
          }
        ]
      }
    ]
  }
}
```

### Feature Roles

| `role` | Meaning | Example |
|---|---|---|
| `path` | Action proceeds along a path | `follow the hallway` |
| `where` | Action occurs at a location | `turn left at the sofa` |
| `progress` | Action passes a landmark | `walk past the table` |
| `target` | Movement target | `go to the door` |
| `terminate` | Task completion condition | `until you see the sofa` |
| `start` | Task start trigger | `when you see the sofa, turn left` |

### Behavior Matrix

| Input type | `parse_instruction_auto` | `parse_instruction` | `parse_instruction_llm` |
|---|---|---|---|
| Simple (1-2 actions, no temporal words) | Rule engine | Rule engine | Rule fallback only on LLM failure |
| Complex (before/after/until/instead of/do not, 3+ actions, stairs/elevator) | LLM with voting + step-level backtracking | `status=needs_review`, empty tasks | LLM with step-level backtracking; rule fallback only on backend failure |
| Active vertical (upstairs, downstairs, take elevator) | `status=unsupported` | `status=unsupported` | `status=unsupported` |

### Step-Level Backtracking

When the LLM parser produces multiple competing interpretations, it now emits **step-level backtracking candidates** instead of full-plan alternatives:

- **`backtracking.step_candidates`** — a list of groups, one per ambiguous step.
- Each group contains `rank: 2|3` candidates that differ from the primary task at that `step_id`.
- The `alternatives` field is deprecated for new output and will be empty when `status="ok"`.

This design lets downstream planners explore local revisions (e.g., "turn left" vs "turn right" at step 2) without re-parsing the entire instruction.

### Running VLN Parser Tests

```bash
python -m pytest vln_instruction_parser/tests -v
```
