import "server-only";
import { NextResponse } from "next/server";

export const json = (body: unknown, status = 200) =>
  NextResponse.json(body, { status, headers: { "Cache-Control": "no-store, private" } });

/**
 * Browsers send an Origin header on POSTs. Refuse ones from other sites
 * (a second layer: the SameSite=Lax session cookie already isn't sent).
 */
export function isSameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  return origin === null || origin === new URL(request.url).origin;
}
