import "server-only";
import { headers } from "next/headers";

/*
 * The only origins Kelsie will send people back to after Google sign-in or
 * an email link. Supabase enforces the same list (Authentication -> URL
 * Configuration -> Redirect URLs); this is a second check in our own code.
 */
export const PRODUCTION_ORIGIN = "https://kelsie-ai-beta.vercel.app";
const ALLOWED_ORIGINS = new Set(["http://localhost:3000", PRODUCTION_ORIGIN]);

/**
 * This deployment's origin if it's on the allowlist, otherwise production.
 * The Host header comes from the client, so it's only trusted when it
 * matches the list exactly.
 */
export async function siteOrigin(): Promise<string> {
  const h = await headers();
  const host = h.get("x-forwarded-host") ?? h.get("host");
  const proto = h.get("x-forwarded-proto") ?? (host?.startsWith("localhost") ? "http" : "https");
  const origin = host ? `${proto}://${host}` : "";
  return ALLOWED_ORIGINS.has(origin) ? origin : PRODUCTION_ORIGIN;
}
