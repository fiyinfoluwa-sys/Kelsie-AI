/*
 * Kelsie's floating orb, injected only on sites the user allowed.
 *
 * Page safety:
 *  - Everything lives in a *closed* Shadow DOM: the website's scripts can't
 *    reach inside it, and the website's CSS can't restyle it.
 *  - Built with createElement/textContent only. No innerHTML, no eval, no
 *    remote code.
 *  - Reads nothing from the page: no text, no form fields, no passwords.
 *  - Holds no secrets. It can only ask the background worker "should I
 *    show here?", "hide me on this site" and "open the profile".
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
    @media (prefers-reduced-motion: reduce) { .orb { transition: none; } }
  `;

  let host = null;
  let orb = null;
  let panel = null;

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

    root.append(panel, orb);

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
    host?.remove();
    host = orb = panel = null;
  }

  /** Asks the background worker whether the orb belongs on this page. */
  async function check() {
    const reply = await send({ type: "kelsie-hello" });
    if (reply?.show) {
      if (!host || !host.isConnected) mount();
    } else {
      teardown();
    }
  }

  chrome.runtime.onMessage.addListener((message, sender) => {
    // Only the extension itself can send these.
    if (sender.id !== chrome.runtime.id) return;
    if (message?.type === "kelsie-recheck") check();
  });

  check();
})();
