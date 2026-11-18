import { NextResponse, type NextRequest } from "next/server";
import { homeFor } from "@/lib/session";
import { createClient } from "@/lib/supabase/server";

/**
 * Google sends people back here with a one-time code. Exchanging it needs
 * the PKCE verifier cookie set when sign-in started, so a code stolen from
 * the URL is useless in another browser.
 *
 * Destinations are fixed paths on this site: there's no "next" parameter
 * that could be abused to redirect somewhere else.
 */
export async function GET(request: NextRequest) {
  const code = request.nextUrl.searchParams.get("code");
  const fail = NextResponse.redirect(new URL("/signin?error=google", request.url));
  if (!code) return fail;

  const supabase = await createClient();
  const { data, error } = await supabase.auth.exchangeCodeForSession(code);
  if (error || !data.user) return fail;

  return NextResponse.redirect(new URL(await homeFor(supabase, data.user.id), request.url));
}
