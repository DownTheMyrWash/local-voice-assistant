from __future__ import annotations

import os
import json
from typing import Any, AsyncIterator

from shared.config import load_config
from shared.logging_utils import get_logger
from .base import Action, ActionContext

log = get_logger("llm")

cfg = load_config()
HISTORY_FILE = cfg.chat_history_file or "chat_history.json"

class ClearHistoryAction(Action):
    name = "clear_history"
    description = (
        "Wipes out the entire conversation history logs, causing you to completely forget "
        "past session contexts. Trigger this only if the user explicitly requests to clear, "
        "wipe, reset, delete, or forget their chat history."
    )
    parameters = {"type": "object", "properties": {}}

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        # We handle clearing disk storage directly via the tool execution scope
        if os.path.exists(HISTORY_FILE):
            try:
                os.remove(HISTORY_FILE)
                yield "I have cleared our conversation history and forgotten past sessions."
            except Exception as e:
                log.error("Failed to delete history file: %s", e)
                yield "I encountered an error trying to clear the history file."
        else:
            yield "Your conversation history is already empty."