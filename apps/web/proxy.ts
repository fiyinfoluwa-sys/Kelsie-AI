import { NextResponse, type NextRequest } from "next/server";
import { updateSession } from "@/lib/supabase/proxy";

// Pages that need a signed-in user.
const PROTECTED = ["/onboarding", "/profile", "/settings", "/reset-password", "/connect-extension"];
// Pages that make no sense once signed in.
const SIGNED_OUT_ONLY = ["/signin", "/signup"];

function matches(pathname: string, prefixes: string[]) {
  return prefixes.some((p) => pathname === p || pathname.startsWith(`${p}/`));
}

export async function proxy(request: NextRequest) {
  const { pathname, searchParams } = request.nextUrl;

  // If Supabase falls back to the bare Site URL for an email link, the
  // token lands on the home page: forward it to the confirm route.
  if (pathname === "/" && (searchParams.has("token_hash") || searchParams.has("code"))) {
    const confirm = new URL("/auth/confirm", request.url);
    for (const key of ["token_hash", "type", "code"]) {
      const value = searchParams.get(key);
      if (value) confirm.searchParams.set(key, value);
    }
    return NextResponse.redirect(confirm);
  }

  const { response, userId } = await updateSession(request);

  // Sign-in pages showing a message (e.g. "verify your email") stay put,
  // so an unverified session can't loop between /signin and /profile.
  const hasMessage = searchParams.has("verify") || searchParams.has("error");

  let destination: string | null = null;
  if (!userId && matches(pathname, PROTECTED)) destination = "/signin";
  if (userId && !hasMessage && matches(pathname, SIGNED_OUT_ONLY)) destination = "/profile";
  if (!destination) return response;

  // Carry over any refreshed session cookies onto the redirect.
  const redirect = NextResponse.redirect(new URL(destination, request.url));
  for (const cookie of response.cookies.getAll()) {
    redirect.cookies.set(cookie);
  }
  return redirect;
}

export const config = {
  matcher: [
    // Everything except static files and images.
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)",
  ],
};
