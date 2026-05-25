"""Quick demo of the instruction_polish API."""

from instruction_polish import polish, analyze, list_strategies, register_strategy

print("=== Available strategies ===")
for s in list_strategies():
    print(f"  - {s}")

samples = [
    "can u help me write python code",
    "Basically i wanna know how neural networks work",
    "In order to succeed, you must work hard",
]

print("\n=== Polish examples ===")
for raw in samples:
    print(f"\nOriginal: {raw!r}")
    for strat in ["default", "clarity", "conciseness"]:
        print(f"  [{strat:12s}] {polish(raw, strategy=strat)!r}")

print("\n=== Strategy chaining ===")
print(polish("u summarize this", strategy="clarity,detail"))

print("\n=== Analysis ===")
for raw in samples:
    result = analyze(raw)
    print(f"{raw!r}: score={result['score']}, reasons={result['score_reasons']}")

print("\n=== Custom strategy ===")
register_strategy("shout", lambda s: s.upper().strip(".?!") + "!!!")
print(polish("hello world", strategy="shout"))
