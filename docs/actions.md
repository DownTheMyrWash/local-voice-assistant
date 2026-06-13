# Actions

Actions are how the assistant does things in the real world — checking the time, opening websites, controlling smart-home devices, etc.

## How it works

1. The LLM is given the list of registered actions (name + description).
2. If the user's request matches an action, the LLM replies with a tool-call payload:
   ```
   <tool_call>{"name": "get_time", "arguments": {}}</tool_call>
   ```
3. The server runs the matching action's `run()` method and TTS's the result.
4. If no action matches, the LLM's plain text reply is TTS'd directly.

## Adding a new action

Create a new file in `pc-server/actions/` (or add to an existing one):

```python
# pc-server/actions/my_actions.py
from typing import Any, AsyncIterator
from .base import Action, ActionContext


class GreetAction(Action):
    name = "greet"
    description = "Greets the user by name if known."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name to greet"}
        },
        "required": ["name"],
    }

    async def run(self, arguments: dict[str, Any], ctx: ActionContext) -> AsyncIterator[str]:
        name = arguments.get("name", "friend")
        yield f"Hey {name}, how's it going?"
```

Then register it in [pc-server/actions/builtins.py](../pc-server/actions/builtins.py):

```python
def register_default_actions() -> None:
    reg = get_registry()
    for cls in (GetTimeAction, OpenYoutubeAction, TellJokeAction, GreetAction):
        if reg.get(cls.name) is None:
            reg.register(cls())
```

That's it. No changes to the pipeline or the LLM code are needed.

## Returning streamed text

`Action.run` is an `AsyncIterator[str]`, so you can yield in chunks. Each chunk becomes TTS input. This is useful for actions that fetch or generate data in pieces.

## Bundled starters

| Name           | Trigger phrases                  | What it does                            |
| -------------- | -------------------------------- | --------------------------------------- |
| `get_time`     | "what time is it"                | Returns the current local time          |
| `open_youtube` | "open youtube"                   | Opens youtube.com in the default browser|
| `tell_joke`    | "tell me a joke"                 | Returns a canned joke                   |
| `stop`         | "stop", "cancel", "never mind"   | Client-side interrupt (no LLM call)     |

## Ideas for future actions

- `get_weather` — needs an API key + city parameter
- `play_music` — control Spotify / local player
- `lights_on` / `lights_off` — smart-home via Home Assistant
- `set_timer` — local timer with audible callback
- `clipboard_read` — read whatever's on the user's clipboard
