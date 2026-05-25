"""Instruction Polish - A tool for optimizing AI instructions."""

__version__ = "0.1.0"

from .core import polish, analyze, list_strategies
from .batch import process_lines, process_file
from .strategies import register_strategy
from .suggest import suggest_strategy

__all__ = [
    "polish", "analyze", "list_strategies",
    "process_lines", "process_file",
    "register_strategy", "suggest_strategy",
]
