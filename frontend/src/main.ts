/**
 * JARVIS — Main entry point.
 *
 * Wires together the orb visualization, WebSocket communication,
 * speech recognition, and audio playback into a single experience.
 */

import { createOrb, type OrbState } from "./orb";
import { createVoiceInput, createAudioPlayer } from "./voice";
import { createSocket } from "./ws";
import { openSettings, checkFirstTimeSetup, SPEECH_LANG_EVENT } from "./settings";
import "./style.css";

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

type State = "idle" | "listening" | "thinking" | "speaking";
let currentState: State = "idle";
let isMuted = false;

const statusEl = document.getElementById("status-text")!;
const errorEl = document.getElementById("error-text")!;
const captionEl = document.getElementById("jarvis-caption")!;

function showError(msg: string) {
  errorEl.textContent = msg;
  errorEl.style.opacity = "1";
  setTimeout(() => {
    errorEl.style.opacity = "0";
  }, 5000);
}

let captionTimer: number | undefined;

/**
 * Put what JARVIS said on screen.
 *
 * Shown whether or not TTS produced audio: with a voice it reads as
 * subtitles, and without one it is the only way to see the reply — the
 * response used to reach nothing but the devtools console.
 */
function showCaption(text: unknown) {
  if (typeof text !== "string" || !text.trim()) return;
  clearTimeout(captionTimer);
  captionEl.textContent = text;
  captionEl.style.opacity = "1";
  // Hold long enough to read it, since a caption may be all the user gets —
  // truer now than when this was written: the Commands panel's "Dispatch
  // status" can return several paragraphs, not the sentence or two a spoken
  // reply is limited to. The 60ms/char rate is ~16.7 chars/s, close to a
  // measured comfortable reading pace; the outer cap is a backstop against
  // something unbounded slipping through, not a real ceiling — at 90s it
  // wouldn't bind before ~1400 characters, well past anything this app sends.
  const holdMs = Math.min(90000, 4000 + text.length * 60);
  captionTimer = window.setTimeout(() => {
    captionEl.style.opacity = "0";
  }, holdMs);
}

function updateStatus(state: State) {
  const labels: Record<State, string> = {
    idle: "",
    listening: "listening...",
    thinking: "thinking...",
    speaking: "",
  };
  statusEl.textContent = labels[state];
}

// ---------------------------------------------------------------------------
// Init components
// ---------------------------------------------------------------------------

const canvas = document.getElementById("orb-canvas") as HTMLCanvasElement;
const orb = createOrb(canvas);

const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${wsProto}//${window.location.host}/ws/voice`;
const socket = createSocket(WS_URL);

const audioPlayer = createAudioPlayer();
orb.setAnalyser(audioPlayer.getAnalyser());

function transition(newState: State) {
  if (newState === currentState) return;
  currentState = newState;
  orb.setState(newState as OrbState);
  updateStatus(newState);

  switch (newState) {
    case "idle":
      if (!isMuted) voiceInput.resume();
      break;
    case "listening":
      if (!isMuted) voiceInput.resume();
      break;
    case "thinking":
      voiceInput.pause();
      break;
    case "speaking":
      voiceInput.pause();
      break;
  }
}

// ---------------------------------------------------------------------------
// Voice input
// ---------------------------------------------------------------------------

/**
 * Fragment accumulation.
 *
 * Chrome closes a recognition segment at every pause, so a sentence spoken
 * with any hesitation arrives as several "final" results. Sending each one
 * immediately made JARVIS answer half a thought and then receive the rest as
 * a separate question. Instead, collect the pieces and send once the speaker
 * has actually stopped.
 */
const UTTERANCE_GAP_MS = 1000;
// A pause never arrives while someone is dictating steadily, so cap the wait
// rather than let a long sentence hold the whole conversation open.
const UTTERANCE_MAX_MS = 6000;

let pendingFragments: string[] = [];
let gapTimer: number | undefined;
let maxTimer: number | undefined;

function discardUtterance() {
  clearTimeout(gapTimer);
  clearTimeout(maxTimer);
  gapTimer = undefined;
  maxTimer = undefined;
  pendingFragments = [];
}

function flushUtterance() {
  clearTimeout(gapTimer);
  clearTimeout(maxTimer);
  gapTimer = undefined;
  maxTimer = undefined;

  const text = pendingFragments.join(" ").replace(/\s+/g, " ").trim();
  pendingFragments = [];
  if (!text) return;

  socket.send({ type: "transcript", text, isFinal: true });
  transition("thinking");
}

const voiceInput = createVoiceInput(
  (text: string) => {
    // Cancel any current JARVIS response before sending new input. Done on the
    // first fragment, not at flush time: cutting him off is what the user
    // wanted the moment they started talking.
    audioPlayer.stop();

    pendingFragments.push(text);
    clearTimeout(gapTimer);
    gapTimer = window.setTimeout(flushUtterance, UTTERANCE_GAP_MS);
    if (maxTimer === undefined) {
      maxTimer = window.setTimeout(flushUtterance, UTTERANCE_MAX_MS);
    }
  },
  (msg: string) => {
    showError(msg);
  }
);

// ---------------------------------------------------------------------------
// Text input — for when speaking is not an option
// ---------------------------------------------------------------------------

/**
 * A typed message travels the same path a spoken one does; the only addition
 * is `silent`, which tells the server to answer in text rather than out loud.
 * Typing usually means someone nearby should not have to hear this.
 *
 * The bar stays hidden until summoned, so the orb keeps the screen to itself.
 */
const textBar = document.getElementById("text-bar")!;
const textInput = document.getElementById("text-input") as HTMLInputElement;

let textBarOpen = false;

function openTextBar(seed = "") {
  if (textBarOpen) return;
  textBarOpen = true;
  textBar.classList.add("visible");
  document.body.classList.add("text-open");

  // The microphone would otherwise keep listening and send whatever it hears
  // while the user is mid-sentence at the keyboard — two inputs racing to
  // answer the same moment.
  discardUtterance();
  voiceInput.pause();

  if (seed) textInput.value = seed;
  textInput.focus();
  updateHint();
}

function closeTextBar() {
  if (!textBarOpen) return;
  textBarOpen = false;
  textBar.classList.remove("visible");
  document.body.classList.remove("text-open");
  textInput.value = "";
  textInput.blur();
  updateHint();

  // Muted means the user switched the microphone off deliberately; closing the
  // bar must not quietly switch it back on.
  if (!isMuted && currentState !== "thinking" && currentState !== "speaking") {
    voiceInput.resume();
  }
}

function updateHint() {
  textBar.classList.toggle("has-text", textInput.value.trim().length > 0);
}

function sendTypedMessage() {
  const text = textInput.value.trim();
  if (!text) return;

  audioPlayer.stop(); // Cut off any reply still playing, as speaking does.
  socket.send({ type: "transcript", text, isFinal: true, silent: true });
  transition("thinking");

  // The bar stays open: someone who typed once is usually still unable to
  // speak, and reopening it for every message would be its own annoyance.
  textInput.value = "";
  updateHint();
}

textInput.addEventListener("input", updateHint);

textInput.addEventListener("keydown", (e) => {
  // Keystrokes inside the bar are the bar's own business — without this they
  // would also reach the global handler below and re-trigger it.
  e.stopPropagation();
  if (e.key === "Enter") {
    e.preventDefault();
    sendTypedMessage();
  } else if (e.key === "Escape") {
    e.preventDefault();
    closeTextBar();
  }
});

/** True when the keystroke belongs to some other field — settings, mainly. */
function isTypingElsewhere(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  const tag = el.tagName.toLowerCase();
  return (
    tag === "input" ||
    tag === "textarea" ||
    tag === "select" ||
    el.isContentEditable
  );
}

/**
 * The settings panel is a modal, so keystrokes over it are meant for it — even
 * the ones that land on the backdrop rather than on a field.
 */
function settingsOpen(): boolean {
  return !!document.getElementById("settings-container")?.classList.contains("open");
}

document.addEventListener("keydown", (e) => {
  // Escape is handled here as well as on the input itself, so the bar can
  // always be dismissed even if focus has drifted somewhere else — otherwise a
  // bar that lost focus would be stuck open with no way back to the orb.
  if (textBarOpen && e.key === "Escape") {
    e.preventDefault();
    closeTextBar();
    return;
  }

  if (textBarOpen || settingsOpen() || isTypingElsewhere(e.target)) return;

  // ⌘K / Ctrl-K summons it deliberately.
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    openTextBar();
    return;
  }

  if (e.metaKey || e.ctrlKey || e.altKey) return;

  // Otherwise just start typing. A single printable character opens the bar
  // and becomes its first character, so the thought is never interrupted by
  // having to find a button first.
  if (e.key.length === 1 && e.key !== " ") {
    e.preventDefault();
    openTextBar(e.key);
  }
});

// ---------------------------------------------------------------------------
// Audio playback finished
// ---------------------------------------------------------------------------

audioPlayer.onFinished(() => {
  transition("idle");
});

// ---------------------------------------------------------------------------
// WebSocket messages
// ---------------------------------------------------------------------------

socket.onMessage((msg) => {
  const type = msg.type as string;

  if (type === "audio") {
    const audioData = msg.data as string;
    console.log("[audio] received", audioData ? `${audioData.length} chars` : "EMPTY", "state:", currentState);
    if (audioData) {
      if (currentState !== "speaking") {
        transition("speaking");
      }
      audioPlayer.enqueue(audioData);
    } else {
      // TTS failed — no audio but still need to return to idle
      console.warn("[audio] no data received, returning to idle");
      transition("idle");
    }
    // Log text for debugging
    if (msg.text) console.log("[JARVIS]", msg.text);
    showCaption(msg.text);
  } else if (type === "status") {
    const state = msg.state as string;
    if (state === "thinking" && currentState !== "thinking") {
      transition("thinking");
    } else if (state === "working") {
      // Task spawned — show thinking with a different label
      transition("thinking");
      statusEl.textContent = "working...";
    } else if (state === "idle") {
      transition("idle");
    }
  } else if (type === "text") {
    // Text fallback when TTS fails
    console.log("[JARVIS]", msg.text);
    showCaption(msg.text);
  } else if (type === "task_spawned") {
    console.log("[task]", "spawned:", msg.task_id, msg.prompt);
  } else if (type === "task_complete") {
    console.log("[task]", "complete:", msg.task_id, msg.status, msg.summary);
  } else if (type === "activity") {
    pushActivity(msg.text as string);
  }
});

// ---------------------------------------------------------------------------
// HUD — system/weather tiles (polled) and the activity feed (pushed)
// ---------------------------------------------------------------------------
// Two different update strategies for two different kinds of data. CPU,
// memory and the weather change on their own clock, indifferent to anything
// JARVIS does — polling every few seconds is the honest model for that, and
// cheap enough not to matter. Activity is the opposite: it only exists at the
// instant something happens, so it has to be pushed, not polled for.

const hudTopLeft = document.getElementById("hud-top-left")!;
const hudWeatherPanel = document.getElementById("hud-weather")!;
const hudWeatherBody = document.getElementById("hud-weather-body")!;
const hudActivityList = document.getElementById("hud-activity-list")!;

const HUD_POLL_MS = 5000;

/** Above this fraction of the gauge, a glance should catch it. */
function gaugeLevel(percent: number): "normal" | "high" | "critical" {
  if (percent >= 90) return "critical";
  if (percent >= 75) return "high";
  return "normal";
}

function setGauge(name: string, percent: number, valueText: string) {
  const el = document.querySelector(`.hud-gauge[data-gauge="${name}"]`) as HTMLElement | null;
  if (!el) return;
  const clamped = Math.max(0, Math.min(100, percent));
  const fill = el.querySelector(".hud-gauge-fill") as HTMLElement;
  const value = el.querySelector(".hud-gauge-value") as HTMLElement;
  fill.style.width = `${clamped}%`;
  value.textContent = valueText;
  const level = gaugeLevel(clamped);
  if (level === "normal") el.removeAttribute("data-level");
  else el.setAttribute("data-level", level);
}

// WMO weather codes collapsed to one glyph — the panel has room for a
// character, not a sentence; the spoken reply already carries the detail.
/**
 * Picks a glyph from `condition_key` (e.g. "weather.mostly_clear"), never
 * from the displayed `condition` text — that text is translated server-side
 * via phrase(), so matching English substrings against it works only in
 * English and silently falls through to the thermometer everywhere else.
 * condition_key is the one field in HudWeather that phrase() never touches.
 */
function weatherEmoji(conditionKey: string): string {
  if (conditionKey.includes("clear")) return "☀️";
  if (conditionKey.includes("cloudy") || conditionKey.includes("overcast")) return "☁️";
  if (conditionKey.includes("fog")) return "🌫️";
  if (conditionKey.includes("thunderstorm")) return "⛈️";
  if (conditionKey.includes("snow")) return "🌨️";
  if (conditionKey.includes("rain") || conditionKey.includes("drizzle") || conditionKey.includes("showers")) return "🌧️";
  return "🌡️";
}

interface HudWeather {
  label: string;
  temp: number;
  unit_symbol: string;
  humidity: number | null;
  condition: string;
  condition_key: string;
}

interface HudStats {
  cpu_percent: number;
  memory_percent: number;
  tokens_today: number;
  tokens_target: number;
  tokens_percent: number;
  weather: HudWeather | null;
}

async function pollHudStats() {
  try {
    const res = await fetch("/api/hud/stats", { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return;
    const stats = (await res.json()) as HudStats;

    setGauge("cpu", stats.cpu_percent, `${Math.round(stats.cpu_percent)}%`);
    setGauge("memory", stats.memory_percent, `${Math.round(stats.memory_percent)}%`);
    setGauge("tokens", stats.tokens_percent, `${Math.round(stats.tokens_percent)}%`);
    hudTopLeft.classList.add("ready");

    if (stats.weather) {
      const w = stats.weather;
      hudWeatherPanel.style.display = "";
      hudWeatherBody.innerHTML = "";
      const tempEl = document.createElement("span");
      tempEl.className = "hud-weather-temp";
      tempEl.textContent = `${weatherEmoji(w.condition_key)} ${Math.round(w.temp)}${w.unit_symbol}`;
      const detailEl = document.createElement("span");
      detailEl.className = "hud-weather-detail";
      detailEl.textContent = w.humidity != null ? `${w.condition}\nUmidità ${w.humidity}%` : w.condition;
      detailEl.style.whiteSpace = "pre-line";
      hudWeatherBody.appendChild(tempEl);
      hudWeatherBody.appendChild(detailEl);
    } else {
      hudWeatherPanel.style.display = "none";
    }
  } catch {
    // A missed poll is invisible by design — the gauges just hold their last
    // reading for another 5s, which reads as calm rather than broken.
  }
}

pollHudStats();
setInterval(pollHudStats, HUD_POLL_MS);

// -- Activity feed --

const HUD_ACTIVITY_MAX = 6;
// Newest first. Each entry ages out of the "fresh" look after a while so the
// feed reads as a log, not a wall of equally-urgent lines.
const HUD_ACTIVITY_STALE_MS = 12000;

interface ActivityEntry {
  text: string;
  ts: number;
}

let activityEntries: ActivityEntry[] = [];

function renderActivity() {
  hudActivityList.innerHTML = "";
  const now = Date.now();
  for (const entry of activityEntries) {
    const line = document.createElement("div");
    line.className = "hud-activity-line";
    if (now - entry.ts > HUD_ACTIVITY_STALE_MS) line.classList.add("stale");
    line.textContent = entry.text;
    hudActivityList.appendChild(line);
  }
}

function pushActivity(text: string) {
  if (!text) return;
  activityEntries.unshift({ text, ts: Date.now() });
  activityEntries = activityEntries.slice(0, HUD_ACTIVITY_MAX);
  renderActivity();
}

// Sweep for staleness independent of new arrivals — a feed that only ever
// re-renders on a new event would freeze mid-fade the moment things go quiet.
setInterval(renderActivity, 3000);

// -- Quick action buttons --

function sendQuickAction(action: string) {
  socket.send({ type: "quick_action", action });
}

document.querySelectorAll<HTMLButtonElement>(".hud-action").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const action = btn.dataset.action;
    if (action === "voice") {
      // Reuse the mic button's own logic rather than re-implement mute —
      // two toggles maintaining the same boolean independently is how they
      // end up disagreeing.
      btnMute.click();
      btn.classList.toggle("muted", isMuted);
    } else if (action === "vision" || action === "terminal") {
      sendQuickAction(action);
    }
  });
});

// Command shortcuts share the exact same quick_action wire format and the
// exact same server-side handler as the action buttons above — the only
// difference is which four actions this row happens to trigger.
document.querySelectorAll<HTMLButtonElement>(".hud-command").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const command = btn.dataset.command;
    if (command) sendQuickAction(command);
  });
});

// -- Status card: uptime (ticked client-side), voice backend, connected tools --

const hudUptimeEl = document.getElementById("hud-uptime")!;
const hudVoiceBackendEl = document.getElementById("hud-voice-backend")!;
const hudConnectedList = document.getElementById("hud-connected-list")!;

function formatUptime(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

interface SettingsStatus {
  claude_code_installed: boolean;
  calendar_accessible: boolean;
  mail_accessible: boolean;
  notes_accessible: boolean;
  vault_accessible: boolean;
  uptime_seconds: number;
  tts_backend: string;
}

const TTS_BACKEND_LABELS: Record<string, string> = {
  fish: "Fish Audio",
  macos: "macOS",
  none: "Assente",
};

/**
 * Fetched once, not polled. Reachability probes here shell out to
 * AppleScript per integration — fine for one page load, not for a 5s loop.
 * Uptime keeps counting between fetches via the client's own clock, seeded
 * from the one real reading the server gave us.
 */
async function loadConnectedStatus() {
  let status: SettingsStatus;
  try {
    const res = await fetch("/api/settings/status", { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return;
    status = (await res.json()) as SettingsStatus;
  } catch {
    return; // Server not ready yet, or a probe genuinely hung — try again never; a stale "—" is honest here.
  }

  hudVoiceBackendEl.textContent = TTS_BACKEND_LABELS[status.tts_backend] ?? status.tts_backend;

  const fetchedAt = Date.now();
  const baseUptime = status.uptime_seconds;
  hudUptimeEl.textContent = formatUptime(baseUptime);
  setInterval(() => {
    hudUptimeEl.textContent = formatUptime(baseUptime + (Date.now() - fetchedAt) / 1000);
  }, 1000);

  const items: Array<[string, boolean]> = [
    ["Calendario", status.calendar_accessible],
    ["Posta", status.mail_accessible],
    ["Note", status.notes_accessible],
    ["Vault", status.vault_accessible],
    ["Browser", true], // Playwright ships with the app; nothing to configure
    ["Claude Code", status.claude_code_installed],
  ];
  hudConnectedList.innerHTML = "";
  for (const [label, on] of items) {
    const row = document.createElement("div");
    row.className = "hud-connected-item" + (on ? "" : " off");
    row.innerHTML = `<span class="hud-connected-dot"></span><span>${label}</span>`;
    hudConnectedList.appendChild(row);
  }
}

loadConnectedStatus();

// ---------------------------------------------------------------------------
// Kick off
// ---------------------------------------------------------------------------

/**
 * Read the speech language chosen in the settings panel.
 *
 * Resolved before the microphone opens so the first session is built in the
 * right language: switching a live session means tearing it down and racing
 * for the microphone, which is worth avoiding on every page load. Bounded, so
 * an unreachable server costs a moment rather than the microphone.
 */
async function fetchSpeechLanguage(): Promise<string | null> {
  try {
    const res = await fetch("/api/settings/preferences", {
      signal: AbortSignal.timeout(2000),
    });
    const prefs = (await res.json()) as { speech_lang?: string };
    return prefs.speech_lang || null;
  } catch {
    return null; // Server not ready — keep the default language.
  }
}

// Start listening after a brief delay for the orb to render
setTimeout(async () => {
  const lang = await fetchSpeechLanguage();
  if (lang) voiceInput.setLanguage(lang);
  voiceInput.start();
  transition("listening");
}, 1000);

// Saving the setting re-tunes the running session — no reload needed.
document.addEventListener(SPEECH_LANG_EVENT, (e) => {
  const lang = (e as CustomEvent<string>).detail;
  if (lang) voiceInput.setLanguage(lang);
});

// Resume AudioContext on ANY user interaction (browser autoplay policy)
function ensureAudioContext() {
  const ctx = audioPlayer.getAnalyser().context as AudioContext;
  if (ctx.state === "suspended") {
    ctx.resume().then(() => console.log("[audio] context resumed"));
  }
}
document.addEventListener("click", ensureAudioContext);
document.addEventListener("touchstart", ensureAudioContext);
document.addEventListener("keydown", ensureAudioContext, { once: true });

// Try to resume audio context on load
ensureAudioContext();

// ---------------------------------------------------------------------------
// UI Controls
// ---------------------------------------------------------------------------

const btnMute = document.getElementById("btn-mute")!;
const btnMenu = document.getElementById("btn-menu")!;
const menuDropdown = document.getElementById("menu-dropdown")!;
const btnRestart = document.getElementById("btn-restart")!;
const btnFixSelf = document.getElementById("btn-fix-self")!;

btnMute.addEventListener("click", (e) => {
  e.stopPropagation();
  isMuted = !isMuted;
  btnMute.classList.toggle("muted", isMuted);
  if (isMuted) {
    // Half-spoken words must not arrive a second after the user silenced him.
    discardUtterance();
    voiceInput.pause();
    transition("idle");
  } else {
    voiceInput.resume();
    transition("listening");
  }
});

btnMenu.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = menuDropdown.style.display === "none" ? "block" : "none";
});

document.addEventListener("click", () => {
  menuDropdown.style.display = "none";
});

btnRestart.addEventListener("click", async (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  statusEl.textContent = "restarting...";
  try {
    await fetch("/api/restart", { method: "POST" });
    // Wait a few seconds then reload
    setTimeout(() => window.location.reload(), 4000);
  } catch {
    statusEl.textContent = "restart failed";
  }
});

btnFixSelf.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  // Activate work mode on the WebSocket session (JARVIS becomes Claude Code's voice)
  socket.send({ type: "fix_self" });
  statusEl.textContent = "entering work mode...";
});

// Settings button
const btnSettings = document.getElementById("btn-settings")!;
btnSettings.addEventListener("click", (e) => {
  e.stopPropagation();
  menuDropdown.style.display = "none";
  openSettings();
});

// First-time setup detection — check after a short delay for server readiness
setTimeout(() => {
  checkFirstTimeSetup();
}, 2000);
