import "server-only";
import { createServerClient } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { cookieOptions, supabaseEnv } from "./config";

/**
 * Refreshes the session cookie if needed and reports who is signed in.
 *
 * Returns the response to send (it may carry refreshed cookies) and the
 * verified user ID, or null when nobody is signed in.
 */
export async function updateSession(request: NextRequest) {
  let response = NextResponse.next({ request });

  // Without Supabase settings, treat everyone as signed out (fail closed)
  // so the public pages still load and protected ones still redirect.
  let env: { url: string; key: string };
  try {
    env = supabaseEnv();
  } catch (error) {
    console.error(error);
    return { response, userId: null };
  }
  const { url, key } = env;

  const supabase = createServerClient(url, key, {
    cookieOptions,
    cookies: {
      getAll() {
        return request.cookies.getAll();
      },
      setAll(cookiesToSet, headers) {
        for (const { name, value } of cookiesToSet) {
          request.cookies.set(name, value);
        }
        response = NextResponse.next({ request });
        for (const { name, value, options } of cookiesToSet) {
          response.cookies.set(name, value, options);
        }
        // Stops CDNs from caching a response that sets session cookies.
        for (const [header, value] of Object.entries(headers ?? {})) {
          response.headers.set(header, value);
        }
      },
    },
  });

  // getClaims() verifies the token's signature, unlike getSession(), which
  // trusts whatever is in the cookie. Don't run code between creating the
  // client and this call.
  const { data } = await supabase.auth.getClaims();
  const userId = typeof data?.claims?.sub === "string" ? data.claims.sub : null;

  return { response, userId };
}
