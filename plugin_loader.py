"""JARVIS plugin loader — drop a file in plugins/, get a tool.

Capabilities used to cost three edits in three places: a paragraph in
JARVIS_SYSTEM_STATIC, a branch in the action dispatcher, and a hope that the
model emits the tag in exactly the right shape. That is why there are eighteen
[ACTION:X] tags and a ~3600-token instruction block: the catalogue only grows.

A plugin costs one file. It declares itself with a PLUGIN dict and a run()
function; discovery finds it at startup and hands Claude a real tool
declaration, so the selection is a validated tool call rather than text the
model has to imitate and a regex has to recover.

This runs ALONGSIDE the [ACTION:X] system, which is untouched. Claude can emit
a tool call and speak in the same turn, so the two coexist and existing
behaviour is unchanged for anyone who adds no plugins.

Writing one — copy plugins/_template.py:

    PLUGIN = {"name": "...", "description": "...", "input_schema": {...}}
    def run(params: dict) -> str: ...

Nothing else in the codebase needs to change.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("jarvis.plugins")

PLUGINS_DIR = Path(__file__).parent / "plugins"
STATE_FILE = Path(__file__).parent / "data" / "plugins.json"

# Tool names Claude sees. Lowercase snake_case keeps them visually distinct
# from the SHOUTED [ACTION:X] tags in the prompt, which matters when reading a
# transcript to work out which of the two systems fired.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_DEFAULT_SCHEMA = {"type": "object", "properties": {}}

# A plugin holds up the voice turn that called it — the user is sitting there
# waiting for a reply. Long jobs should return "on it, sir" immediately and
# finish in a background task, the way the lookup system already does.
DEFAULT_TIMEOUT = 30.0

# How often a plugin's optional check() runs. Five minutes by default because
# the things worth interrupting someone over do not change by the second, and
# several of these reach for AppleScript, which is not free.
DEFAULT_CHECK_INTERVAL = 300.0


@dataclass
class PluginRecord:
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=lambda: dict(_DEFAULT_SCHEMA))
    run: Optional[Callable] = None
    timeout: float = DEFAULT_TIMEOUT
    # Optional second entry point: a plugin may also watch for something worth
    # saying unprompted. See proactive.py — the plugin decides WHAT is worth
    # saying, the service decides WHETHER it is a decent moment to say it.
    check: Optional[Callable] = None
    check_interval: float = DEFAULT_CHECK_INTERVAL
    file: str = ""
    valid: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Enable/disable state
# ---------------------------------------------------------------------------

def _enabled_state() -> dict:
    """Read on every call, never cached — toggling a plugin takes effect on the
    next voice turn instead of at the next restart."""
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def is_enabled(name: str) -> bool:
    """Plugins are on unless explicitly switched off: dropping a file in is
    already the deliberate act, so asking for a second one is just a way to
    make new plugins look broken."""
    return _enabled_state().get(name, True)


def set_enabled(name: str, enabled: bool) -> None:
    state = _enabled_state()
    state[name] = bool(enabled)
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.warning(f"Could not persist plugin state: {e}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PluginRegistry:
    def __init__(self, plugins: dict[str, PluginRecord]):
        self._plugins = plugins                     # valid entries only
        self._all: list[PluginRecord] = []          # valid + rejected, for the UI

    def tool_declarations(self) -> list[dict]:
        """Anthropic tool blocks for every enabled plugin. A disabled plugin is
        not merely refused at call time — the model never learns it exists."""
        return [
            {
                "name": rec.name,
                "description": rec.description,
                "input_schema": rec.input_schema,
            }
            for rec in self._plugins.values()
            if is_enabled(rec.name)
        ]

    def has(self, name: str) -> bool:
        return name in self._plugins

    async def run(self, name: str, params: dict, ws=None, client=None) -> str:
        """Execute a plugin and return the string that goes back to Claude as
        the tool result. Never raises: a broken plugin degrades into a spoken
        apology, it does not take down the voice turn."""
        rec = self._plugins.get(name)
        if rec is None or not rec.valid:
            return f"Plugin '{name}' is not available."
        if not is_enabled(name):
            return f"The '{name}' plugin is currently disabled."

        try:
            result = await asyncio.wait_for(
                self._invoke(rec, params, ws, client), timeout=rec.timeout
            )
            return str(result) if result else "Done."
        except asyncio.TimeoutError:
            log.warning(f"Plugin '{name}' timed out after {rec.timeout}s")
            return f"The '{name}' plugin took too long and was stopped."
        except Exception as e:
            log.error(f"Plugin '{name}' crashed: {e}")
            log.debug(traceback.format_exc())
            return f"The '{name}' plugin failed: {e}"

    async def _invoke(self, rec: PluginRecord, params: dict, ws, client):
        return await self._call(rec.run, (params,), ws, client)

    async def _call(self, fn: Callable, args: tuple, ws, client):
        """Call a plugin entry point with only the arguments it declares.

        A plugin author writes `def run(params)` and it works; one who needs
        the WebSocket adds `ws` to the signature and gets it, one that has to
        call the model adds `client`. `client` was added after three plugins
        already existed and not one of them needed touching — that is the
        point of doing it this way rather than with a fixed signature.
        """
        sig = inspect.signature(fn)
        takes_all = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        kwargs = {}
        if takes_all or "ws" in sig.parameters:
            kwargs["ws"] = ws
        if takes_all or "client" in sig.parameters:
            kwargs["client"] = client

        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        # A synchronous plugin runs in a worker thread. On the event loop it
        # would stall the audio the user is currently listening to — the loop
        # is serving speech, not just this request.
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))

    def watchers(self) -> list[PluginRecord]:
        """Enabled plugins that also watch for something worth saying."""
        return [
            rec for rec in self._plugins.values()
            if rec.check is not None and is_enabled(rec.name)
        ]

    async def run_check(self, rec: PluginRecord, client=None):
        """Run one plugin's check(). Returns whatever it produced, or None.

        Errors are swallowed to a log line: a watcher that throws every five
        minutes must not become five minutes of noise, and it must never take
        down the proactive loop for the plugins that do work.
        """
        try:
            return await asyncio.wait_for(
                self._call(rec.check, (), None, client), timeout=rec.timeout
            )
        except asyncio.TimeoutError:
            log.warning(f"Plugin '{rec.name}' check timed out")
        except Exception as e:
            log.warning(f"Plugin '{rec.name}' check failed: {e}")
            log.debug(traceback.format_exc())
        return None

    def list_for_ui(self) -> list[dict]:
        return [
            {
                "name": rec.name,
                "description": rec.description,
                "file": rec.file,
                "valid": rec.valid,
                "error": rec.error,
                "enabled": is_enabled(rec.name) if rec.valid else False,
            }
            for rec in self._all
        ]

    def summary(self) -> str:
        ok = len(self._plugins)
        return f"{ok} active, {len(self._all) - ok} rejected"


# ---------------------------------------------------------------------------
# Validation and discovery
# ---------------------------------------------------------------------------

def _validate(module, filename: str) -> PluginRecord:
    """Return a record; .valid=False with .error set on any problem. Never raises."""
    meta = getattr(module, "PLUGIN", None)
    if not isinstance(meta, dict):
        return PluginRecord(name=Path(filename).stem, file=filename,
                            error="missing PLUGIN dict")

    name = meta.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return PluginRecord(name=str(name or Path(filename).stem), file=filename,
                            error="PLUGIN['name'] must be lowercase snake_case "
                                  "starting with a letter")

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        return PluginRecord(name=name, file=filename,
                            error="PLUGIN['description'] missing or empty")

    schema = meta.get("input_schema", _DEFAULT_SCHEMA)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return PluginRecord(name=name, file=filename,
                            error="PLUGIN['input_schema'] must be a JSON Schema "
                                  'object with "type": "object"')

    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        return PluginRecord(name=name, file=filename,
                            error="missing callable run(params, ...)")

    try:
        timeout = float(meta.get("timeout", DEFAULT_TIMEOUT))
        check_interval = float(meta.get("check_interval", DEFAULT_CHECK_INTERVAL))
    except (TypeError, ValueError):
        return PluginRecord(name=name, file=filename,
                            error="PLUGIN['timeout'] and ['check_interval'] must be numbers")

    check_fn = getattr(module, "check", None)
    if check_fn is not None and not callable(check_fn):
        return PluginRecord(name=name, file=filename,
                            error="check must be a function if defined")

    return PluginRecord(name=name, description=description.strip(),
                        input_schema=schema, run=run_fn, timeout=timeout,
                        check=check_fn, check_interval=check_interval,
                        file=filename, valid=True)


def discover_plugins(plugins_dir: Path = PLUGINS_DIR,
                     reserved: set[str] | None = None) -> PluginRegistry:
    """Scan plugins_dir for *.py files and build the registry.

    Files starting with '_' are skipped, which covers __init__.py, _template.py
    and any shared helper an author wants to keep out of the catalogue.

    Nothing in here raises. A plugin with a syntax error, a bad PLUGIN dict or a
    name someone else already took is rejected, logged, and shown in the UI with
    its reason — while every other plugin still loads. One broken file must
    never be able to stop JARVIS from starting.
    """
    reserved = reserved or set()
    valid: dict[str, PluginRecord] = {}
    every: list[PluginRecord] = []

    plugins_dir.mkdir(parents=True, exist_ok=True)

    # Sorted, so which file wins a name collision is deterministic rather than
    # a function of filesystem order.
    for path in sorted(plugins_dir.glob("*.py"), key=lambda p: p.name):
        if path.name.startswith("_"):
            continue
        try:
            module_name = f"plugins.{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("could not build import spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise

            rec = _validate(module, path.name)

            if rec.valid and rec.name in reserved:
                rec = PluginRecord(name=rec.name, file=path.name,
                                   error=f"'{rec.name}' is a reserved name")
            elif rec.valid and rec.name in valid:
                rec = PluginRecord(name=rec.name, file=path.name,
                                   error=f"'{rec.name}' already claimed by "
                                         f"{valid[rec.name].file}")
        except Exception as e:
            rec = PluginRecord(name=path.stem, file=path.name,
                               error=f"failed to load: {e}")
            log.debug(traceback.format_exc())

        every.append(rec)
        if rec.valid:
            valid[rec.name] = rec
            log.info(f"Plugin loaded: {rec.name} ({path.name})")
        else:
            log.warning(f"Plugin rejected: {path.name} — {rec.error}")

    registry = PluginRegistry(valid)
    registry._all = every
    log.info(f"Plugin discovery: {registry.summary()}")
    return registry
