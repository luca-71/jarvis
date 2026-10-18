"""Today's headlines, from Google News RSS.

Named `news`, not `research`, because [ACTION:RESEARCH] still exists and does
something else: it dispatches Claude Code to browse the web for minutes and
write a report document. That is the right tool for "find me every competitor
in the cemetery software market". It is the wrong one for "che notizie ci
sono", where the honest answer takes one second and fits in two spoken
sentences — and where, on the fallback provider, an action tag would not fire
at all.

RSS on purpose: no API key, no quota, no account, ~0.4s, and the source
publication comes with each headline so JARVIS can say where something was
reported rather than stating it as fact.

This returns headlines, not articles. It does not open the pages and cannot
tell you what an article says beyond its title — for that, browse it.
"""

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

log = logging.getLogger("jarvis.plugins.news")

# Language/region for the feed. Google wants all three and they must agree.
_LOCALE = os.getenv("JARVIS_NEWS_LOCALE", "it")
_REGION = os.getenv("JARVIS_NEWS_REGION", "IT")

_TOP = f"https://news.google.com/rss?hl={_LOCALE}&gl={_REGION}&ceid={_REGION}:{_LOCALE}"
_SEARCH = ("https://news.google.com/rss/search?q={q}"
           f"&hl={_LOCALE}&gl={_REGION}&ceid={_REGION}:{_LOCALE}")

# Enough for the model to pick out what matters, few enough that the tool
# result does not become the largest thing in the prompt.
_MAX_HEADLINES = 7

PLUGIN = {
    "name": "news",
    "description": (
        "Current news headlines, optionally about a topic. Use for 'che "
        "notizie ci sono', 'novità su X'. Returns headlines and their source "
        "only — summarise them, never invent detail beyond the title. For a "
        "deep researched report use [ACTION:RESEARCH]."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string",
                      "description": "Subject to search. Omit for general headlines."},
        },
        "required": [],
    },
    "timeout": 25.0,
}

# Google appends " - Publication" to every title, and the publication is
# already a separate field.
_TRAILING_SOURCE = re.compile(r"\s+-\s+[^-]{2,40}$")


def _clean(title: str, source: str) -> str:
    title = " ".join(title.split())
    if source and title.endswith(f"- {source}"):
        return title[: -(len(source) + 2)].strip()
    return _TRAILING_SOURCE.sub("", title).strip() or title


async def run(params: dict) -> str:
    topic = (params.get("topic") or "").strip()
    url = _SEARCH.format(q=quote(topic)) if topic else _TOP

    try:
        async with httpx.AsyncClient(
            timeout=15, follow_redirects=True,
            # Google serves an error page to a bare client.
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        ) as client:
            response = await client.get(url)
    except asyncio.TimeoutError:
        return "The news feed did not respond."
    except Exception as e:
        log.warning(f"News fetch failed: {e}")
        return "I could not reach the news."

    if response.status_code != 200:
        log.warning(f"News feed returned {response.status_code}")
        return "The news feed is not answering."

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        log.warning(f"News feed unparseable: {e}")
        return "The news feed sent something I could not read."

    headlines = []
    for item in root.findall(".//item")[:_MAX_HEADLINES]:
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        clean = _clean(title, source)
        headlines.append(f"{clean} ({source})" if source else clean)

    if not headlines:
        return (f"Nothing came back about {topic}." if topic
                else "No headlines came back.")

    label = f"Headlines about {topic}" if topic else "Today's headlines"
    log.info(f"News: {len(headlines)} headlines{f' about {topic}' if topic else ''}")
    return f"{label}:\n" + "\n".join(f"- {h}" for h in headlines)
