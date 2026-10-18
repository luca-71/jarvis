"""Open a URL or run a web search in the browser.

Converted from [ACTION:BROWSE]. The tag version only ever worked on Claude:
given tool declarations, Gemini treats them as the only way to act and never
writes a literal tag, so on the fallback path "cerca su google il meteo" got
answered with "Cerco subito le previsioni meteo" and nothing opened. As a tool
it works on both.

Being a tool also removes the guesswork. The tag carried a free-text target
that the handler had to pick apart with a regex — deciding whether "joetmd.com"
was a domain or a search phrase, and repairing domains that speech-to-text had
split into two words. Here the model says which of the two it meant.
"""

import logging
from urllib.parse import quote

from actions import open_browser

log = logging.getLogger("jarvis.plugins.browse")

PLUGIN = {
    "name": "browse",
    "description": (
        "Open a web page or search in the browser. Not for opening desktop "
        "apps (open_app), not for questions you can answer yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Address or bare domain, if a site was named."},
            "query": {"type": "string", "description": "Search terms otherwise."},
            "browser": {"type": "string", "enum": ["chrome", "firefox"],
                        "description": "Default chrome."},
        },
        "required": [],
    },
    "timeout": 20.0,
}


async def run(params: dict) -> str:
    url = (params.get("url") or "").strip()
    query = (params.get("query") or "").strip()
    browser = (params.get("browser") or "chrome").strip().lower()

    if not url and not query:
        return "Nothing to open — no address and no search terms."

    if url:
        target = url if url.startswith(("http://", "https://")) else f"https://{url}"
        label = url
    else:
        target = f"https://www.google.com/search?q={quote(query)}"
        label = f"a search for {query}"

    result = await open_browser(target, browser)
    if not result.get("success"):
        log.warning(f"browse failed for {target}")
        return result.get("confirmation") or "The browser would not open that."

    log.info(f"Opened {target} in {browser}")
    return f"Opened {label}."
