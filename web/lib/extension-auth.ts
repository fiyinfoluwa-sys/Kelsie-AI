import "server-only";
import { createHash, randomBytes } from "node:crypto";
import { createClient } from "@supabase/supabase-js";
import { NextResponse } from "next/server";
import { EXTENSION_ID } from "@/lib/extension";
import { supabaseEnv } from "@/lib/supabase/config";

/*
 * Helpers for the extension API (/api/extension/*).
 *
 * The extension never gets a Supabase login. It holds its own random
 * session token; the database stores only the token's SHA-256 hash. The
 * token is 256 bits of randomness, so a plain hash is enough (there's
 * nothing to brute-force, unlike a password).
 */

/** A new 256-bit random secret (connection code or session token). */
export function newSecret(): string {
  return randomBytes(32).toString("base64url");
}

export function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

/** Codes and tokens are always 43 base64url characters. */
export function isSecretShape(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9_-]{43}$/.test(value);
}

/** The token from "Authorization: Bearer <token>", if well-formed. */
export function bearerToken(request: Request): string | null {
  const header = request.headers.get("authorization") ?? "";
  const match = /^Bearer ([A-Za-z0-9_-]{43})$/.exec(header);
  return match ? match[1] : null;
}

/**
 * A Supabase client with no user session and no cookies. It can only call
 * the extension_* database functions, which check the token themselves.
 */
export function anonClient() {
  const { url, key } = supabaseEnv();
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
  });
}

/*
 * Only Kelsie's own extension may call this API from a browser. CORS
 * allows exactly its origin, so no website can read the responses, and
 * the extension needs no permission for this site. (The token in each
 * request is what actually proves who's calling.)
 */
const EXTENSION_ORIGIN = `chrome-extension://${EXTENSION_ID}`;

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": EXTENSION_ORIGIN,
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Authorization, Content-Type",
  "Access-Control-Max-Age": "600",
  Vary: "Origin",
};

/** Requests from a browser must come from Kelsie's extension. */
export function fromAllowedOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  return origin === null || origin === EXTENSION_ORIGIN;
}

export function json(body: unknown, status = 200) {
  return NextResponse.json(body, {
    status,
    headers: { ...CORS_HEADERS, "Cache-Control": "no-store, private" },
  });
}

/** CORS preflight for the extension. */
export function preflight(request: Request) {
  if (!fromAllowedOrigin(request)) return new NextResponse(null, { status: 403 });
  return new NextResponse(null, { status: 204, headers: CORS_HEADERS });
}

export const unauthorized = () => json({ error: "not_connected" }, 401);
export const forbidden = () => json({ error: "forbidden" }, 403);
