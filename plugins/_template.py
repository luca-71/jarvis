"""Drop-in JARVIS plugin template.

Copy this file, rename it without the leading underscore, fill in PLUGIN and
run(). Nothing else needs to change — JARVIS finds it at startup and Claude
gets it as a real tool.

Files starting with '_' are skipped by the loader, so this template never
appears in the catalogue, and you can keep shared helpers in _something.py.
"""

PLUGIN = {
    # Lowercase snake_case, unique. This is the name Claude calls.
    "name": "my_plugin",

    # This is the only thing Claude reads when deciding whether to call you, so
    # write it for that job: say what it does, when to use it, and — if it
    # could be confused with another tool — which one to use instead. A vague
    # description is the single most common reason a working plugin never fires.
    "description": (
        "One or two sentences. Be explicit about the phrases that should "
        "trigger it, and about what it does NOT cover."
    ),

    # Standard JSON Schema. Claude fills these in and the API validates them
    # before your code ever runs, so you can trust what arrives in params.
    "input_schema": {
        "type": "object",
        "properties": {
            "example_arg": {
                "type": "string",
                "description": "What this argument means",
            },
        },
        "required": [],
    },

    # Optional, defaults to 30s. The user is waiting through this, so keep it
    # short: for anything slower, return "on it, sir" now and finish in a
    # background task.
    "timeout": 30.0,
}


def run(params: dict) -> str:
    """Return a short string — Claude receives it and speaks it in his own words.

    The signature is introspected: `def run(params)` is enough, and adding `ws`
    gets you the WebSocket for sending things to the frontend directly. May be
    `async def` — a plain def runs in a worker thread so it cannot stall the
    audio that is playing.

    Errors are caught by the loader, but catching your own gets the user a
    better sentence than a stack trace's last line.
    """
    example_arg = params.get("example_arg", "")
    return f"Did the thing with {example_arg}."
