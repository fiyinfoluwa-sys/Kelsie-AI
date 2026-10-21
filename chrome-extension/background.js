/*
 * Kelsie's background service worker.
 *
 * The only part of the extension that holds the session token or talks to
 * the Kelsie website. Content scripts (which run inside web pages) never
 * see the token; they can only ask "should I show the orb here?".
 */

import { SITE_ORIGIN, SITE_URL } from "./config.js";
import { SITES, isSiteId, patternsFor, urlMatches } from "./sites.js";

const SCRIPT_ID = "kelsie-orb";
const SYNC_ALARM = "kelsie-sync";
const CONNECT_WINDOW_MS = 10 * 60 * 1000;
const VIBES = new Set(["chill", "hype", "direct"]);
// "chrome-extension://<id>/": every page that belongs to this extension.
const EXTENSION_PREFIX = chrome.runtime.getURL("");

/*
 * Storage:
 *  - chrome.storage.local: token, account, hiddenHosts. Only extension
 *    pages and this worker read it; the content script never does.
 *  - chrome.storage.session: the one-time "state" for a pending connect.
 *    Cleared when the browser closes.
 */

// ------------------------------------------------------------
// Talking to the Kelsie website
// ------------------------------------------------------------

async function api(path, { method = "GET", body, token } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  return fetch(`${SITE_URL}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    // No cookies, no cache, no referrer: the token is the only credential.
    credentials: "omit",
    cache: "no-store",
    referrerPolicy: "no-referrer",
  });
}

/** Accepts only the exact shape we expect from the server. */
function parseAccount(data) {
  if (typeof data !== "object" || data === null) return null;
  const sites = Array.isArray(data.allowed_sites) ? data.allowed_sites.filter(isSiteId) : [];
  return {
    name: typeof data.name === "string" ? data.name.slice(0, 40) : "",
    vibe: VIBES.has(data.vibe) ? data.vibe : null,
    allowedSites: sites,
    memoryEnabled: data.memory_enabled !== false,
  };
}

/**
 * Re-reads the account from the server and applies it. A 401 means the
 * session was revoked (disconnected, signed out on the website, or the
 * account was deleted), so the extension logs itself out.
 */
async function refresh() {
  const { token } = await chrome.storage.local.get("token");
  if (!token) {
    await applySites([]);
    return null;
  }

  let response;
  try {
    response = await api("/api/extension/state", { token });
  } catch {
    // Offline: keep what we had.
    const { account } = await chrome.storage.local.get("account");
    return account ?? null;
  }

  if (response.status === 401) {
    await logOut();
    return null;
  }
  if (!response.ok) {
    const { account } = await chrome.storage.local.get("account");
    return account ?? null;
  }

  const account = parseAccount(await response.json().catch(() => null));
  if (!account) return null;
  await chrome.storage.local.set({ account });
  await applySites(account.allowedSites);
  return account;
}

/** Forgets the session and stops running everywhere. */
async function logOut() {
  await chrome.storage.local.remove(["token", "account"]);
  await applySites([]);
}

// ------------------------------------------------------------
// Where the orb runs
// ------------------------------------------------------------

async function grantedPatterns(patterns) {
  const granted = [];
  for (const origin of patterns) {
    if (await chrome.permissions.contains({ origins: [origin] })) granted.push(origin);
  }
  return granted;
}

/**
 * Makes the extension run on exactly the allowed sites it has permission
 * for, and nowhere else:
 *  - drops Chrome permission for any site no longer allowed,
 *  - registers the content script only for allowed + granted addresses,
 *  - tells every open tab to re-check, so the orb disappears right away
 *    where it's no longer allowed.
 */
async function applySites(allowedSiteIds) {
  const wanted = patternsFor(allowedSiteIds);

  const { origins = [] } = await chrome.permissions.getAll();
  const stale = origins.filter((origin) => !wanted.includes(origin));
  if (stale.length) {
    await chrome.permissions.remove({ origins: stale }).catch(() => {});
  }

  const active = await grantedPatterns(wanted);
  const previous = (await chrome.scripting.getRegisteredContentScripts({ ids: [SCRIPT_ID] }))[0];
  const previousMatches = previous?.matches ?? [];

  if (active.length === 0) {
    if (previous) await chrome.scripting.unregisterContentScripts({ ids: [SCRIPT_ID] });
  } else {
    const script = {
      id: SCRIPT_ID,
      matches: active,
      js: ["content.js"],
      runAt: "document_idle",
      allFrames: false,
      persistAcrossSessions: true,
    };
    if (previous) await chrome.scripting.updateContentScripts([script]);
    else await chrome.scripting.registerContentScripts([script]);
  }

  await chrome.storage.local.set({ activePatterns: active });
  await broadcastRecheck();

  // Newly allowed sites: show the orb in tabs that are already open there.
  const added = active.filter((p) => !previousMatches.includes(p));
  if (added.length) await injectIntoOpenTabs();
}

async function broadcastRecheck() {
  const tabs = await chrome.tabs.query({});
  await Promise.all(
    tabs.map((tab) =>
      tab.id === undefined
        ? null
        : chrome.tabs.sendMessage(tab.id, { type: "kelsie-recheck" }).catch(() => {}),
    ),
  );
}

async function injectIntoOpenTabs() {
  const tabs = await chrome.tabs.query({});
  await Promise.all(
    tabs.map((tab) =>
      tab.id === undefined
        ? null
        : // Fails quietly on any tab Kelsie has no permission for.
          chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] }).catch(() => {}),
    ),
  );
}

/** Should the orb show on this page? Decided here, not in the page. */
async function shouldShow(url) {
  const { token, account, activePatterns = [], hiddenHosts = [] } = await chrome.storage.local.get([
    "token",
    "account",
    "activePatterns",
    "hiddenHosts",
  ]);
  if (!token || !account || !url) return false;
  if (!activePatterns.some((pattern) => urlMatches(pattern, url))) return false;
  return !hiddenHosts.includes(new URL(url).hostname);
}

/** Allowed sites the user still needs to grant Chrome permission for. */
async function sitesMissingPermission(allowedSiteIds) {
  const missing = [];
  for (const id of allowedSiteIds) {
    const site = SITES.find((s) => s.id === id);
    if (site && (await grantedPatterns(site.patterns)).length < site.patterns.length) {
      missing.push({ id: site.id, label: site.label });
    }
  }
  return missing;
}

// ------------------------------------------------------------
// Connecting
// ------------------------------------------------------------

function randomState() {
  const bytes = crypto.getRandomValues(new Uint8Array(24));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** Opens the website's connect page with a one-time state value. */
async function startConnect() {
  const state = randomState();
  await chrome.storage.session.set({ pendingConnect: { state, expires: Date.now() + CONNECT_WINDOW_MS } });
  await chrome.tabs.create({ url: `${SITE_URL}/connect-extension?state=${encodeURIComponent(state)}` });
}

/**
 * Handles the code the website sends after the user clicks "Connect".
 * Only accepted if the extension started this connect (matching state)
 * within the last 10 minutes.
 */
async function finishConnect(code, state) {
  const { pendingConnect } = await chrome.storage.session.get("pendingConnect");
  if (!pendingConnect || pendingConnect.state !== state || Date.now() > pendingConnect.expires) {
    return { ok: false, error: "This connect link expired. Click Connect in the Kelsie extension again." };
  }
  if (typeof code !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(code)) {
    return { ok: false, error: "Something went wrong. Try again from the extension." };
  }
  // One use only, whatever happens next.
  await chrome.storage.session.remove("pendingConnect");

  let response;
  try {
    response = await api("/api/extension/exchange", { method: "POST", body: { code } });
  } catch {
    return { ok: false, error: "Couldn't reach Kelsie. Check your connection and try again." };
  }
  const data = await response.json().catch(() => null);
  if (!response.ok || typeof data?.token !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(data.token)) {
    return { ok: false, error: "That code didn't work. Try again from the extension." };
  }

  await chrome.storage.local.set({ token: data.token });
  const account = await refresh();
  if (account && (await sitesMissingPermission(account.allowedSites)).length) {
    // Asking for site access needs a click from the user, so open a page
    // with the "Allow" button.
    await chrome.tabs.create({ url: chrome.runtime.getURL("popup.html?view=tab") });
  }
  return { ok: true };
}

// ------------------------------------------------------------
// Messages
// ------------------------------------------------------------

/** Messages from the Kelsie website (externally_connectable). */
chrome.runtime.onMessageExternal.addListener((message, sender, sendResponse) => {
  // Chrome only delivers messages from sites listed in the manifest; this
  // narrows it to the one site in config.js.
  if (sender.origin !== SITE_ORIGIN) return false;

  if (message?.type === "kelsie-connect") {
    finishConnect(message.code, message.state).then(sendResponse);
    return true;
  }
  if (message?.type === "kelsie-sync") {
    // Carries no data: just re-read everything from the server.
    refresh().finally(() => sendResponse({ ok: true }));
    return true;
  }
  return false;
});

/** Messages from inside the extension: the popup and the content script. */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id) return false;
  const fromExtensionPage = typeof sender.url === "string" && sender.url.startsWith(EXTENSION_PREFIX);

  // From the content script in a web page: limited to these three.
  if (!fromExtensionPage && sender.tab) {
    const pageUrl = sender.url ?? sender.tab.url;
    switch (message?.type) {
      case "kelsie-hello":
        shouldShow(pageUrl).then((show) => sendResponse({ show }));
        return true;
      case "kelsie-hide":
        hideOn(pageUrl).then(() => sendResponse({ ok: true }));
        return true;
      case "kelsie-open-profile":
        chrome.tabs.create({ url: `${SITE_URL}/profile` }).then(() => sendResponse({ ok: true }));
        return true;
      default:
        return false;
    }
  }

  // Everything else only from the extension's own pages (the popup).
  if (!fromExtensionPage) return false;
  handlePopup(message).then(sendResponse, () => sendResponse({ ok: false }));
  return true;
});

async function hideOn(url) {
  // The address comes from Chrome (sender.url), not from the page.
  let host;
  try {
    host = new URL(url).hostname;
  } catch {
    return;
  }
  const { hiddenHosts = [] } = await chrome.storage.local.get("hiddenHosts");
  if (!hiddenHosts.includes(host)) {
    await chrome.storage.local.set({ hiddenHosts: [...hiddenHosts, host].slice(-200) });
  }
  await broadcastRecheck();
}

async function popupState() {
  const { token, hiddenHosts = [] } = await chrome.storage.local.get(["token", "hiddenHosts"]);
  if (!token) return { connected: false };
  const account = await refresh();
  if (!account) return { connected: false };
  return {
    connected: true,
    account,
    missing: await sitesMissingPermission(account.allowedSites),
    hiddenCount: hiddenHosts.length,
    siteUrl: SITE_URL,
  };
}

async function handlePopup(message) {
  switch (message?.type) {
    case "popup-state":
      return popupState();

    case "popup-connect":
      await startConnect();
      return { ok: true };

    case "popup-permissions-changed":
      await refresh();
      return popupState();

    case "popup-pause": {
      const { token } = await chrome.storage.local.get("token");
      if (!token || typeof message.paused !== "boolean") return { ok: false };
      const response = await api("/api/extension/memory", {
        method: "POST",
        body: { enabled: !message.paused },
        token,
      }).catch(() => null);
      if (response?.status === 401) await logOut();
      return { ok: Boolean(response?.ok), state: await popupState() };
    }

    case "popup-unhide-all":
      await chrome.storage.local.set({ hiddenHosts: [] });
      await broadcastRecheck();
      return popupState();

    case "popup-open-profile":
      await chrome.tabs.create({ url: `${SITE_URL}/profile` });
      return { ok: true };

    case "popup-disconnect": {
      const { token } = await chrome.storage.local.get("token");
      if (token) {
        // Deletes the session on the server. Logging out locally happens
        // even if this fails, and a dead token would be rejected anyway.
        await api("/api/extension/disconnect", { method: "POST", token }).catch(() => {});
      }
      await logOut();
      return { connected: false };
    }

    default:
      return { ok: false };
  }
}

// ------------------------------------------------------------
// Staying in sync
// ------------------------------------------------------------

async function ensureAlarm() {
  if (!(await chrome.alarms.get(SYNC_ALARM))) {
    // Catches website changes even if no Kelsie page is open: removed
    // sites, sign-out or account deletion take effect within a minute.
    await chrome.alarms.create(SYNC_ALARM, { periodInMinutes: 1 });
  }
}

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === SYNC_ALARM) refresh();
});

chrome.runtime.onInstalled.addListener(() => {
  ensureAlarm();
  refresh();
});

chrome.runtime.onStartup.addListener(() => {
  ensureAlarm();
  refresh();
});

// If the user changes Kelsie's site access in Chrome's own settings.
chrome.permissions.onAdded.addListener(() => refresh());
chrome.permissions.onRemoved.addListener(() => refresh());
