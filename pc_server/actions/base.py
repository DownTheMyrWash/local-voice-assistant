"""Action base class and registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class ActionContext:
    """Runtime context passed to actions (Pi client info, settings, etc.)."""
    client_addr: str = "unknown"


class Action:
    """Base class for voice-assistant actions.

    Subclass and implement `name`, `description`, `parameters`, and `run`.
    Register with `get_registry().register(MyAction())`.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        """Yield text chunks (will be TTS'd). Override this."""
        raise NotImplementedError
        yield ""  # pragma: no cover  (makes this a generator for type checkers)


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        if not action.name:
            raise ValueError("Action.name is required")
        if action.name in self._actions:
            raise ValueError(f"Duplicate action name: {action.name}")
        self._actions[action.name] = action

    def get(self, name: str) -> Action | None:
        return self._actions.get(name)

    def tool_specs(self) -> list[dict[str, Any]]:
        """Return the tool list in Ollama's expected format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": a.name,
                    "description": a.description,
                    "parameters": a.parameters or {"type": "object", "properties": {}},
                },
            }
            for a in self._actions.values()
        ]

    def descriptions(self) -> str:
        return "\n".join(
            f"- {a.name}: {a.description}" for a in self._actions.values()
        )


_REGISTRY: ActionRegistry | None = None


def get_registry() -> ActionRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ActionRegistry()
    return _REGISTRY
