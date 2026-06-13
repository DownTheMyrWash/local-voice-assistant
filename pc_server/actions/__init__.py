"""Action plugin system.

Add a new action by creating a file in this directory that defines a
subclass of `Action`, then appending it to `register_default_actions`
or to a custom registry.

The LLM is told which actions are available and is expected to reply
with a JSON tool-call payload when it wants to invoke one. Otherwise it
replies with plain text that gets TTS'd.
"""
from __future__ import annotations

from .base import Action, ActionContext, ActionRegistry, get_registry
from .builtins import register_default_actions

# Auto-register the starter actions on import.
register_default_actions()

__all__ = ["Action", "ActionContext", "ActionRegistry", "get_registry"]
