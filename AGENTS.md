# Agent Guide: Instruction Polish

## Project Purpose

`instruction_polish` is a standalone Python package for polishing and optimizing AI instructions (prompts). It provides rule-based strategies, heuristic quality scoring, batch processing, and an extensible strategy registry.

## Key Files

| File | Role |
|------|------|
| `src/instruction_polish/core.py` | Main API: `polish()`, `analyze()`, `list_strategies()` |
| `src/instruction_polish/strategies.py` | Built-in strategies and `register_strategy()` API |
| `src/instruction_polish/scoring.py` | Heuristic quality scoring (0-100) |
| `src/instruction_polish/batch.py` | File-based batch processing |
| `src/instruction_polish/cli.py` | CLI entry point (`instruction-polish`) |
| `tests/` | pytest unit tests |

## Common Tasks

### Add a new built-in strategy
1. Implement the strategy function in `strategies.py`.
2. Register it via `register_strategy(name, func)` at module level.
3. Add tests in `tests/test_strategies.py`.
4. Update this file and `README.md` if documenting.

### Modify scoring rules
1. Edit `src/instruction_polish/scoring.py`.
2. Update `tests/test_scoring.py` with new expectations.
3. Ensure scores stay in range 0-100.

### Extend the CLI
1. Edit `src/instruction_polish/cli.py`.
2. Keep backward compatibility for existing arguments.

## Build / Test

- Install: `pip install -e .`
- Test: `python -m pytest tests/ -v`
- No external runtime dependencies beyond the standard library.

## Conventions

- Python 3.8+
- Type hints preferred.
- Use the logger from `utils.setup_logging()` rather than raw `print` in library code.
- Strategies must be pure functions: `str -> str`.
