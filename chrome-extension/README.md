# Kelsie for Chrome

Version 1: connects to your Kelsie account and shows a small grey orb on the sites you allowed in Kelsie's Settings. No memory or deadline detection yet.

## Install (Load unpacked)

1. Unzip `kelsie-extension-<version>.zip` (if you were sent one) somewhere you'll keep it.
2. Open `chrome://extensions` and turn on **Developer mode** (top right).
3. Click **Load unpacked** and pick the folder with `manifest.json` in it.
4. Pin Kelsie: click the puzzle-piece icon in the toolbar, then the pin next to Kelsie.
5. Click the Kelsie icon → **Connect to Kelsie**, then **Connect this browser** on the page that opens.
6. Click **Allow** when Kelsie asks for the sites you picked.

## Settings

`config.js` holds the only setting: which Kelsie site to talk to (`https://kelsie-ai-beta.vercel.app` or `http://localhost:3000`). Reload the extension on `chrome://extensions` after changing it.

## Sharing

`sh scripts/package.sh` creates `dist/kelsie-extension-<version>.zip`. It refuses to run while `config.js` points at localhost.

## How it keeps you safe

- **No access at install.** Kelsie asks Chrome for access only to the sites you allowed in Settings, one prompt at a time, and drops access as soon as you remove a site.
- **Runs only where you said.** The orb's script is registered for allowed sites only, and removed from open tabs the moment a site is removed.
- **Sees nothing on the page.** The orb lives in a closed Shadow DOM, reads no page content, and never touches form fields, passwords or payment details.
- **No secrets in the extension.** It holds a random session token, stored by the background worker only. The website keeps just a hash of it, and checks it on every request.
- **Disconnects for real.** Disconnecting, signing out on the website, or deleting your account revokes the token; the extension notices within a minute (instantly if a Kelsie page is open) and stops everywhere.
