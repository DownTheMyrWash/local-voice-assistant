"""Starter actions. Each one is a self-contained file -> easy to copy/edit.

Add a new action:
    1. Copy any file here
    2. Change the class name, `name`, `description`, `parameters`
    3. Implement `run`
    4. Register it in `register_default_actions` (or call the registry directly)
"""
from __future__ import annotations

import webbrowser
from datetime import datetime
from typing import Any, AsyncIterator

from .base import Action, ActionContext, get_registry


class GetTimeAction(Action):
    name = "get_time"
    description = "Tells the user the current local time."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        now = datetime.now().strftime("%I:%M %p").lstrip("0")
        yield f"It's {now}."


class OpenYoutubeAction(Action):
    name = "open_youtube"
    description = "Opens YouTube in the default browser."
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        webbrowser.open("https://www.youtube.com")
        yield "Opening YouTube."


class TellJokeAction(Action):
    name = "tell_joke"
    description = "Tells a short joke. No parameters needed."
    parameters = {"type": "object", "properties": {}}

    _JOKES = [
        "Why don't scientists trust atoms? Because they make up everything.",
        "I told my computer I needed a break. Now it won't stop sending me vacation ads.",
        "Why did the developer go broke? Because he used up all his cache.",
    ]

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        # Deterministic but cheap variety
        idx = datetime.now().microsecond % len(self._JOKES)
        yield self._JOKES[idx]


from .spotify import register_spotify_actions
from .clear_history import ClearHistoryAction

def register_default_actions() -> None:
    reg = get_registry()
    
    for cls in (GetTimeAction, OpenYoutubeAction, TellJokeAction, ClearHistoryAction):
        action = cls()  # Instantiate first
        if reg.get(action.name) is None:
            reg.register(action)
            
    register_spotify_actions()