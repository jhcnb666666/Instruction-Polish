"""Context resolution across steps.

This module previously inherited landmarks from previous steps, but per the
pruning specification we no longer auto-fill missing locations.
"""

from typing import Dict, Any, List


def resolve_context_for_candidates(
    candidates: List[Dict[str, Any]], context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    No-op context resolver.

    Does NOT inherit landmarks from previous steps, as the compact schema
    requires that every feature be explicitly stated in the instruction.
    """
    return candidates


def update_context(context: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return an empty context so that later steps do not inherit locations.
    """
    return {}
