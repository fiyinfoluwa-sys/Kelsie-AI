/*
 * The Kelsie Chrome extension's ID. It's fixed by the public key in
 * chrome-extension/manifest.json, so it's the same on every computer.
 * Not a secret: it only tells the website which extension to talk to.
 */
export const EXTENSION_ID = "lmblhhbijfihdomgiknimkdjndijljac";

/** Asks the extension (if installed) to re-check the account right away. */
export function notifyExtension(): void {
  if (typeof window === "undefined") return;
  const runtime = (window as unknown as { chrome?: { runtime?: ChromeRuntime } }).chrome?.runtime;
  // chrome.runtime only exists here if an installed extension allows this
  // site to message it (externally_connectable).
  if (!runtime?.sendMessage) return;
  try {
    runtime.sendMessage(EXTENSION_ID, { type: "kelsie-sync" }, () => {
      // Reading lastError silences "no receiving end" when it isn't installed.
      void runtime.lastError;
    });
  } catch {
    // Extension not installed or disabled.
  }
}

export type ChromeRuntime = {
  sendMessage: (id: string, message: unknown, callback?: (response: unknown) => void) => void;
  lastError?: { message?: string };
};
