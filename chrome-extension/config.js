/*
 * Which Kelsie website the extension talks to. This is the only setting.
 *
 *   Live site:        "https://kelsie-ai-beta.vercel.app"
 *   Your computer:    "http://localhost:3000"
 *
 * After changing it, click the reload icon for Kelsie on
 * chrome://extensions. Messages from any other site are ignored.
 */
export const SITE_URL = "https://kelsie-ai-beta.vercel.app";

export const SITE_ORIGIN = new URL(SITE_URL).origin;
