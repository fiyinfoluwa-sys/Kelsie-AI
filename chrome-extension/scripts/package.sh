#!/bin/sh
# Builds dist/kelsie-extension-<version>.zip for sharing with friends
# ("Load unpacked" after unzipping). Only the files the extension needs go
# in: no scripts, no dotfiles.
#
# Usage (from anywhere): sh chrome-extension/scripts/package.sh
set -eu

cd "$(dirname "$0")/.."
VERSION=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' manifest.json | head -1)
OUT_DIR="../dist"
OUT="$OUT_DIR/kelsie-extension-$VERSION.zip"

# Refuse to package a build that points at your own computer.
if grep -q 'SITE_URL = "http://localhost' config.js; then
  echo "config.js points at localhost. Switch SITE_URL to the live site first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$OUT"
zip -q -X -r "$OUT" \
  manifest.json config.js sites.js background.js content.js \
  popup.html popup.css popup.js \
  icons fonts README.md \
  -x '*.DS_Store'

echo "Created $(cd "$OUT_DIR" && pwd)/$(basename "$OUT")"
