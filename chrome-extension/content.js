/*
 * Kelsie's floating orb, injected only on sites the user allowed.
 *
 * Page safety:
 *  - Everything lives in a *closed* Shadow DOM: the website's scripts can't
 *    reach inside it, and the website's CSS can't restyle it.
 *  - Built with createElement/textContent only. No innerHTML, no eval, no
 *    remote code.
 *  - Holds no secrets. It can only talk to the background worker, which
 *    decides what's allowed using the address Chrome reports.
 *
 * Memory (only where the background says read: true):
 *  - After 30 seconds of real activity on a page (tab visible and focused,
 *    and you scrolled, typed or moved the mouse in the last minute), it
 *    takes the page's main readable text once and hands it to the
 *    background worker. Activity is tracked as timestamps only; which
 *    keys you press is never looked at.
 *  - Never on pages with password or payment fields.
 *  - The text comes from an offline copy of the page with every form
 *    field, editable area and hidden element removed first, then
 *    Mozilla Readability picks out the article. Capped at 6,000 characters.
 */
(() => {
  // The script can be injected twice (registered + injected into an open
  // tab); only the first copy runs.
  if (globalThis.__kelsieOrbLoaded) return;
  globalThis.__kelsieOrbLoaded = true;

  const CSS = `
    :host {
      all: initial;
      position: fixed !important;
      right: 20px !important;
      bottom: 20px !important;
      z-index: 2147483647 !important;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      color: #17161c;
    }
    /* Text styles inherit through the shadow boundary, so set them on
       everything inside instead of inheriting from the website. */
    * {
      box-sizing: border-box;
      color: #17161c;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      font-style: normal;
      letter-spacing: normal;
      text-transform: none;
      text-shadow: none;
    }
    .orb {
      display: block;
      width: 48px;
      height: 48px;
      padding: 0;
      border: 0;
      border-radius: 50%;
      cursor: pointer;
      background: radial-gradient(circle at 32% 28%, #ffffff 0%, #e6e5ea 18%, #bebdc6 48%, #92919b 78%, #73727c 100%);
      box-shadow: 0 0 18px rgba(160, 158, 172, 0.65), 0 6px 16px rgba(23, 22, 28, 0.18),
        inset -4px -6px 10px rgba(40, 38, 48, 0.25);
      transition: transform 0.15s ease;
    }
    .orb:hover { transform: scale(1.06); }
    .orb.pulse { animation: pulse 1.4s ease-out 2; }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(107, 78, 255, 0.45), 0 6px 16px rgba(23, 22, 28, 0.18); }
      100% { box-shadow: 0 0 0 18px rgba(107, 78, 255, 0), 0 6px 16px rgba(23, 22, 28, 0.18); }
    }
    .toast {
      position: absolute;
      right: 0;
      bottom: 60px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: #17161c;
      box-shadow: 0 8px 20px rgba(23, 22, 28, 0.2);
      white-space: nowrap;
      font-size: 13px;
    }
    .toast[hidden] { display: none; }
    .toast, .toast * { color: #f7f6f2; }
    .toast button {
      padding: 0;
      border: 0;
      background: none;
      font: inherit;
      font-weight: 700;
      color: #b8a9ff;
      text-decoration: underline;
      cursor: pointer;
    }
    .orb:focus-visible { outline: 3px solid #6b4eff; outline-offset: 3px; }
    .panel {
      position: absolute;
      right: 0;
      bottom: 60px;
      width: 260px;
      padding: 16px;
      border: 1px solid #e4e1ec;
      border-radius: 18px;
      background: #ffffff;
      box-shadow: 0 12px 32px rgba(23, 22, 28, 0.16);
      font-size: 14px;
      line-height: 1.45;
    }
    .panel[hidden] { display: none; }
    .head { display: flex; align-items: center; gap: 8px; margin: 0 0 8px; font-weight: 700; font-size: 15px; }
    .dot {
      width: 14px; height: 14px; border-radius: 50%;
      background: radial-gradient(circle at 32% 28%, #fff 0%, #d9d8de 35%, #8f8e98 100%);
    }
    .text { margin: 0 0 14px; color: #5c5a66; }
    .actions { display: flex; flex-direction: column; gap: 8px; }
    button.action {
      width: 100%;
      min-height: 38px;
      border: 0;
      border-radius: 999px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    .primary { background: #6b4eff; color: #fff; }
    .primary:hover { background: #5a3df0; }
    .secondary { background: transparent; color: #17161c; box-shadow: inset 0 0 0 1.5px #e4e1ec; }
    .secondary:hover { box-shadow: inset 0 0 0 1.5px #17161c; }
    button.action:focus-visible { outline: 3px solid #6b4eff; outline-offset: 2px; }
    @media (prefers-reduced-motion: reduce) { .orb { transition: none; } .orb.pulse { animation: none; } }
  `;

  let host = null;
  let orb = null;
  let panel = null;
  let toast = null;
  let toastTimer = 0;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function applyStyles(root) {
    // Constructed stylesheets aren't blocked by a site's Content Security
    // Policy the way <style> tags can be; fall back to a <style> just in case.
    try {
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(CSS);
      root.adoptedStyleSheets = [sheet];
    } catch {
      root.append(element("style", null, CSS));
    }
  }

  function send(message) {
    return chrome.runtime.sendMessage(message).catch(() => null);
  }

  function togglePanel(open = panel.hidden) {
    panel.hidden = !open;
    orb.setAttribute("aria-expanded", String(open));
    if (open) panel.querySelector("button")?.focus();
  }

  function mount() {
    // A custom tag name, so the site's CSS rules for div etc. don't apply.
    host = document.createElement("kelsie-orb-root");
    const root = host.attachShadow({ mode: "closed" });
    applyStyles(root);

    orb = element("button", "orb");
    orb.type = "button";
    orb.setAttribute("aria-label", "Kelsie");
    orb.setAttribute("aria-expanded", "false");
    orb.addEventListener("click", () => togglePanel());

    panel = element("div", "panel");
    panel.hidden = true;
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Kelsie");

    const head = element("p", "head");
    head.append(element("span", "dot"), document.createTextNode("Kelsie"));
    const text = element("p", "text", "Kelsie's here. Nothing to flag yet.");

    const actions = element("div", "actions");
    const profile = element("button", "action primary", "Open my profile");
    profile.type = "button";
    profile.addEventListener("click", () => {
      togglePanel(false);
      send({ type: "kelsie-open-profile" });
    });
    const hide = element("button", "action secondary", "Hide on this site");
    hide.type = "button";
    hide.addEventListener("click", async () => {
      await send({ type: "kelsie-hide" });
      teardown();
    });
    actions.append(profile, hide);
    panel.append(head, text, actions);

    toast = element("div", "toast");
    toast.hidden = true;
    toast.setAttribute("role", "status");

    root.append(panel, toast, orb);

    root.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) {
        togglePanel(false);
        orb.focus();
      }
    });
    document.addEventListener("click", onOutsideClick, true);

    (document.body ?? document.documentElement).append(host);
  }

  function onOutsideClick(event) {
    if (panel && !panel.hidden && !event.composedPath().includes(host)) togglePanel(false);
  }

  function teardown() {
    document.removeEventListener("click", onOutsideClick, true);
    window.clearTimeout(toastTimer);
    host?.remove();
    host = orb = panel = toast = null;
  }

  // ------------------------------------------------------------
  // "Saved a note · Undo"
  // ------------------------------------------------------------

  function showToast(text, withUndo) {
    if (!toast) return;
    window.clearTimeout(toastTimer);
    toast.replaceChildren(document.createTextNode(text));
    if (withUndo) {
      const undo = element("button", null, "Undo");
      undo.type = "button";
      undo.addEventListener("click", async () => {
        window.clearTimeout(toastTimer);
        const reply = await send({ type: "kelsie-undo" });
        showToast(reply?.undone ? "Note removed" : "Couldn't undo", false);
      });
      toast.append(document.createTextNode(" · "), undo);
    }
    toast.hidden = false;
    toastTimer = window.setTimeout(() => {
      if (toast) toast.hidden = true;
    }, withUndo ? 6000 : 2500);
  }

  function celebrateSave() {
    if (!orb) return;
    orb.classList.remove("pulse");
    void orb.offsetWidth; // restart the animation
    orb.classList.add("pulse");
    if (panel) panel.hidden = true;
    showToast("Saved a note", true);
  }

  // ------------------------------------------------------------
  // Memory: 30 active seconds, then one note per page
  // ------------------------------------------------------------

  const ACTIVE_SECONDS_NEEDED = 30;
  const IDLE_AFTER_MS = 60_000;
  const MAX_TEXT = 6000;
  const MAX_ELEMENTS = 20_000;

  let reading = false;
  let tick = 0;
  let activeSeconds = 0;
  // No credit until the person actually interacts with the page: a tab
  // that's open but untouched never produces a note.
  let lastActivity = 0;
  let pageUrl = location.href;
  let notedThisPage = false;

  // Only *when* something happened is recorded, never what was typed.
  const noteActivity = () => {
    lastActivity = Date.now();
  };
  const ACTIVITY_EVENTS = ["scroll", "wheel", "keydown", "pointerdown", "pointermove", "touchstart"];

  function startReading() {
    if (reading) return;
    reading = true;
    for (const type of ACTIVITY_EVENTS) {
      window.addEventListener(type, noteActivity, { capture: true, passive: true });
    }
    tick = window.setInterval(onTick, 1000);
  }

  function stopReading() {
    if (!reading) return;
    reading = false;
    for (const type of ACTIVITY_EVENTS) window.removeEventListener(type, noteActivity, { capture: true });
    window.clearInterval(tick);
  }

  function onTick() {
    // Single-page apps change the address without reloading.
    if (location.href !== pageUrl) {
      pageUrl = location.href;
      activeSeconds = 0;
      notedThisPage = false;
    }
    if (notedThisPage) return;
    const active =
      document.visibilityState === "visible" &&
      document.hasFocus() &&
      Date.now() - lastActivity < IDLE_AFTER_MS;
    if (!active) return;
    activeSeconds += 1;
    if (activeSeconds >= ACTIVE_SECONDS_NEEDED) {
      notedThisPage = true;
      takeNote();
    }
  }

  /** Pages where Kelsie never reads anything. */
  function hasSensitiveFields() {
    return Boolean(
      document.querySelector(
        [
          'input[type="password"]',
          'input[autocomplete^="cc-"]',
          'input[autocomplete="current-password"]',
          'input[autocomplete="new-password"]',
          'input[autocomplete="one-time-code"]',
          'input[name*="card" i]',
          'input[name*="cvv" i]',
          'input[name*="cvc" i]',
          'input[id*="card" i]',
          'iframe[src*="stripe" i]',
          'iframe[src*="paypal" i]',
          'iframe[src*="braintree" i]',
          'iframe[src*="adyen" i]',
        ].join(","),
      ),
    );
  }

  function isVisible(el) {
    if (typeof el.checkVisibility === "function") {
      return el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true });
    }
    const style = getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
  }

  // Removed from the offline copy before anything is read.
  const NEVER_READ = [
    "input", "textarea", "select", "option", "button", "label",
    '[contenteditable]:not([contenteditable="false"])', '[role="textbox"]', '[role="searchbox"]',
    "[hidden]", '[aria-hidden="true"]',
    "script", "style", "noscript", "template", "iframe", "object", "embed", "svg", "canvas",
    "nav", "footer", "aside", "kelsie-orb-root",
  ].join(",");

  /** The page's main readable text, or null. Reads an offline copy only. */
  function extractReadable() {
    const liveRoot = document.body;
    if (!liveRoot || typeof Readability !== "function") return null;

    // Copy the page (the copy has no window: nothing in it runs or loads),
    // then drop everything that's hidden on the real page.
    const copy = document.cloneNode(true);
    const liveWalker = document.createTreeWalker(liveRoot, NodeFilter.SHOW_ELEMENT);
    const copyWalker = copy.createTreeWalker(copy.body, NodeFilter.SHOW_ELEMENT);
    const hidden = [];
    let live = liveWalker.currentNode;
    let copied = copyWalker.currentNode;
    for (let i = 0; live && copied && i < MAX_ELEMENTS; i += 1) {
      if (!isVisible(live)) hidden.push(copied);
      live = liveWalker.nextNode();
      copied = copyWalker.nextNode();
    }
    for (const node of hidden) node.remove();
    for (const node of copy.querySelectorAll(NEVER_READ)) node.remove();

    const article = new Readability(copy, { charThreshold: 300 }).parse();
    const text = article?.textContent?.replace(/\s+/g, " ").trim();
    if (!text) return null;
    return {
      title: (document.title || article.title || "").trim().slice(0, 200),
      text: text.slice(0, MAX_TEXT),
    };
  }

  async function takeNote() {
    // Checked again right before reading, in case the page changed.
    if (hasSensitiveFields()) return;
    let page;
    try {
      page = extractReadable();
    } catch {
      return;
    }
    if (!page || page.text.length < 200) return;
    const reply = await send({ type: "kelsie-page-text", title: page.title, text: page.text });
    // The text isn't kept here either.
    page = null;
    if (reply?.saved) celebrateSave();
  }

  /** Asks the background worker whether the orb belongs on this page. */
  async function check() {
    const reply = await send({ type: "kelsie-hello" });
    if (reply?.show) {
      if (!host || !host.isConnected) mount();
    } else {
      teardown();
    }
    if (reply?.show && reply.read) startReading();
    else stopReading();
  }

  chrome.runtime.onMessage.addListener((message, sender) => {
    // Only the extension itself can send these.
    if (sender.id !== chrome.runtime.id) return;
    if (message?.type === "kelsie-recheck") check();
  });

  check();
})();
