import "server-only";
import type { CookieOptionsWithName } from "@supabase/ssr";

/*
 * Supabase settings shared by every server-side client.
 *
 * The browser never talks to Supabase: every auth and database call runs on
 * the server. That lets the session cookies be httpOnly (unreadable by page
 * JavaScript, so an XSS bug can't steal a session) and means no Supabase key
 * is ever sent to the browser.
 */

export function supabaseEnv(): { url: string; key: string } {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_PUBLISHABLE_KEY;
  if (!url || !key) {
    throw new Error(
      "Missing SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY. Copy web/.env.example to web/.env.local and fill them in.",
    );
  }
  return { url, key };
}

export const cookieOptions: CookieOptionsWithName = {
  httpOnly: true,
  // Only sent over HTTPS in production. Plain http://localhost still works
  // in development.
  secure: process.env.NODE_ENV === "production",
  // Lax keeps the cookie off cross-site requests (CSRF protection) while
  // still allowing the redirect back from Google to carry it.
  sameSite: "lax",
  path: "/",
};
