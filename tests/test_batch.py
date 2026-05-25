"""Tests for batch processing."""

import json
from pathlib import Path

from instruction_polish.batch import process_lines, process_file


def test_process_lines():
    lines = ["Hello world", "", "Tell me a joke"]
    results = process_lines(lines, strategy="default")
    assert len(results) == 2
    assert results[0]["original"] == "Hello world"
    assert results[0]["polished"] == "Hello world."


def test_process_file(tmp_path):
    input_file = tmp_path / "instructions.txt"
    output_file = tmp_path / "out.json"
    input_file.write_text("line one\nline two\n", encoding="utf-8")

    results = process_file(str(input_file), str(output_file), strategy="clarity")
    assert len(results) == 2
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data) == 2


def test_process_jsonl(tmp_path):
    input_file = tmp_path / "instructions.jsonl"
    output_file = tmp_path / "out.jsonl"
    input_file.write_text(
        '{"instruction": "hello world"}\n{"instruction": "u summarize this"}\n',
        encoding="utf-8",
    )

    results = process_file(str(input_file), str(output_file), strategy="clarity")
    assert len(results) == 2
    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["polished"] == "Hello world."
