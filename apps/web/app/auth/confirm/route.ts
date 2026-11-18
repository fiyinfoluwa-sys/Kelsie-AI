import type { EmailOtpType } from "@supabase/supabase-js";
import { NextResponse, type NextRequest } from "next/server";
import { createClient } from "@/lib/supabase/server";

// Link types we accept, and where each one goes once verified.
const DESTINATIONS: Partial<Record<EmailOtpType, string>> = {
  email: "/onboarding",
  signup: "/onboarding",
  recovery: "/reset-password",
};

/**
 * Verification and password-reset emails land here.
 *
 * - Supabase's default templates send a one-time `code`. Exchanging it
 *   needs the PKCE cookie from the browser that started sign-up or the
 *   reset, so a code copied into another browser is useless.
 * - Custom templates (once custom SMTP is set up) can send a single-use
 *   `token_hash` instead, which also works across browsers.
 *
 * Destinations are fixed paths on this site: nothing in the link can send
 * someone elsewhere.
 */
export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const supabase = await createClient();
  const to = (path: string) => NextResponse.redirect(new URL(path, request.url));

  const code = params.get("code");
  if (code) {
    const { data, error } = await supabase.auth.exchangeCodeForSession(code);
    if (error || !data.session) return to("/signin?error=link");
    // The session records how it was created: a reset link, or sign-up.
    const { data: claimsData } = await supabase.auth.getClaims();
    const amr = claimsData?.claims?.amr;
    const isRecovery =
      Array.isArray(amr) &&
      amr.some((entry) => (entry as { method?: string } | null)?.method === "recovery");
    return to(isRecovery ? "/reset-password" : "/onboarding");
  }

  const tokenHash = params.get("token_hash");
  const type = params.get("type") as EmailOtpType | null;
  const destination = type ? DESTINATIONS[type] : undefined;
  if (!tokenHash || !type || !destination) return to("/signin?error=link");

  const { error } = await supabase.auth.verifyOtp({ type, token_hash: tokenHash });
  if (error) return to(type === "recovery" ? "/forgot-password?expired=1" : "/signin?error=link");

  return to(destination);
}
