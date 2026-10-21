import "server-only";

/**
 * Google sign-in is shown only once it's set up in Google Cloud and
 * Supabase. Turn it on with GOOGLE_SIGN_IN=on.
 */
export function googleSignInEnabled(): boolean {
  return process.env.GOOGLE_SIGN_IN === "on";
}
