/*
 * The toolbar popup. Also used as a full tab ("popup.html?view=tab") right
 * after connecting, to ask for site access.
 *
 * It never sees the session token: every action goes through the
 * background worker.
 */

import { SITES } from "./sites.js";

const VIBE_LABELS = { chill: "Chill vibe", hype: "Hype vibe", direct: "Straight to the point" };
const $ = (id) => document.getElementById(id);

let current = null;

function show(view) {
  for (const id of ["loading", "disconnected", "connected"]) $(id).hidden = id !== view;
}

function showError(text) {
  $("error").textContent = text ?? "";
  $("error").hidden = !text;
}

function ask(message) {
  return chrome.runtime.sendMessage(message);
}

function render(state) {
  current = state;
  showError(null);
  if (!state?.connected) {
    show("disconnected");
    return;
  }
  show("connected");
  const { account, missing = [], hiddenCount = 0 } = state;

  $("name").textContent = account.name || "Connected";
  $("vibe").textContent = VIBE_LABELS[account.vibe] ?? "";

  const running = account.allowedSites.length - missing.length;
  const count = account.allowedSites.length;
  const live = count > 0 && running > 0;
  $("status-dot").className = live ? "dot live" : "dot";
  $("status").textContent =
    count === 0
      ? "Off on every site"
      : `Riding along on ${running} site${running === 1 ? "" : "s"}`;

  $("permission").hidden = missing.length === 0;
  if (missing.length) {
    const names = missing.map((m) => m.label).join(", ");
    $("permission-text").textContent = `Chrome needs your OK before Kelsie can run on: ${names}.`;
    $("allow").textContent = missing.length === 1 ? "Allow" : `Allow all ${missing.length}`;
  }

  $("pause").textContent = account.memoryEnabled ? "Pause" : "Paused · Resume";
  $("unhide").hidden = hiddenCount === 0;
  $("unhide").textContent = `Show Kelsie again on ${hiddenCount} hidden site${hiddenCount === 1 ? "" : "s"}`;
}

async function load() {
  show("loading");
  try {
    render(await ask({ type: "popup-state" }));
  } catch {
    show("disconnected");
    showError("Couldn't load your account. Try again.");
  }
}

$("connect").addEventListener("click", async () => {
  $("connect").disabled = true;
  await ask({ type: "popup-connect" });
  $("connect-note").hidden = false;
});

$("allow").addEventListener("click", async () => {
  const ids = (current?.missing ?? []).map((m) => m.id);
  const origins = [...new Set(SITES.filter((s) => ids.includes(s.id)).flatMap((s) => s.patterns))];
  // Must be called straight from the click: Chrome shows its own prompt
  // listing exactly these sites.
  const granted = await chrome.permissions.request({ origins });
  if (!granted) {
    showError("No problem. Kelsie won't run on those sites until you allow them.");
    return;
  }
  render(await ask({ type: "popup-permissions-changed" }));
});

$("pause").addEventListener("click", async () => {
  $("pause").disabled = true;
  const result = await ask({ type: "popup-pause", paused: current.account.memoryEnabled });
  $("pause").disabled = false;
  if (result?.state) render(result.state);
  if (!result?.ok) showError("That didn't save. Try again.");
});

$("profile").addEventListener("click", async () => {
  await ask({ type: "popup-open-profile" });
  if (!document.body.classList.contains("tab")) window.close();
});

$("unhide").addEventListener("click", async () => {
  render(await ask({ type: "popup-unhide-all" }));
});

$("disconnect").addEventListener("click", async () => {
  $("disconnect").disabled = true;
  render(await ask({ type: "popup-disconnect" }));
  $("disconnect").disabled = false;
});

if (new URLSearchParams(location.search).get("view") === "tab") {
  document.body.classList.add("tab");
}
load();
