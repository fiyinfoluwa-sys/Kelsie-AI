"use server";

/*
 * Auth Server Actions. They run only on the server; the browser just posts
 * the form. Next.js rejects Server Action calls whose Origin doesn't match
 * the site, which protects these from cross-site request forgery.
 */

import { redirect } from "next/navigation";
import {
  MESSAGES,
  isRateLimited,
  normalizeEmail,
  passwordProblem,
  signInMessage,
} from "@/lib/auth-messages";
import { cameFromResetLink, homeFor } from "@/lib/session";
import { siteOrigin } from "@/lib/site";
import { createClient } from "@/lib/supabase/server";

export type AuthFormState = {
  error?: string;
  /** Set after sign-up or a reset request: show the "check your inbox" screen. */
  sentTo?: string;
};

export async function signUp(_prev: AuthFormState, form: FormData): Promise<AuthFormState> {
  const email = normalizeEmail(form.get("email"));
  if (!email) return { error: MESSAGES.badEmail };
  const problem = passwordProblem(form.get("password"));
  if (problem) return { error: problem };

  const supabase = await createClient();
  const { data, error } = await supabase.auth.signUp({
    email,
    password: form.get("password") as string,
    // The verification email links back to our own /auth/confirm route.
    options: { emailRedirectTo: `${await siteOrigin()}/auth/confirm` },
  });

  if (error) {
    if (isRateLimited(error)) return { error: MESSAGES.tooMany };
    if (error.code === "weak_password") return { error: MESSAGES.weakPassword };
    // "user_already_exists" only appears if email confirmation is turned
    // off. Show the usual screen so it doesn't reveal the account exists.
    if (error.code === "user_already_exists") return { sentTo: email };
    return { error: MESSAGES.generic };
  }

  // Only happens if email confirmation is disabled in Supabase.
  if (data.session) redirect("/onboarding");

  // With confirmation on, Supabase answers the same way whether or not the
  // email is already registered, so this screen gives nothing away.
  return { sentTo: email };
}

export async function signIn(_prev: AuthFormState, form: FormData): Promise<AuthFormState> {
  const email = normalizeEmail(form.get("email"));
  const password = form.get("password");
  if (!email || typeof password !== "string" || password.length === 0) {
    return { error: MESSAGES.wrongCredentials };
  }

  const supabase = await createClient();
  const { data, error } = await supabase.auth.signInWithPassword({ email, password });
  if (error || !data.user) return { error: error ? signInMessage(error) : MESSAGES.generic };

  redirect(await homeFor(supabase, data.user.id));
}

export async function signInWithGoogle(): Promise<void> {
  const supabase = await createClient();
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    // Only ever our own allowlisted origin. Supabase checks it again
    // against the Redirect URLs list in the dashboard.
    options: { redirectTo: `${await siteOrigin()}/auth/callback` },
  });
  if (error || !data.url) redirect("/signin?error=google");
  redirect(data.url);
}

export async function requestPasswordReset(
  _prev: AuthFormState,
  form: FormData,
): Promise<AuthFormState> {
  const email = normalizeEmail(form.get("email"));
  if (!email) return { error: MESSAGES.badEmail };

  const supabase = await createClient();
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${await siteOrigin()}/auth/confirm`,
  });
  if (error && isRateLimited(error)) return { error: MESSAGES.tooMany };

  // Same response whether or not the account exists.
  return { sentTo: email };
}

export async function updatePassword(_prev: AuthFormState, form: FormData): Promise<AuthFormState> {
  const password = form.get("password");
  const problem = passwordProblem(password);
  if (problem) return { error: problem };
  if (password !== form.get("confirm")) return { error: MESSAGES.passwordsDiffer };

  const supabase = await createClient();
  const { data: claimsData } = await supabase.auth.getClaims();
  const claims = claimsData?.claims;
  if (!claims || !cameFromResetLink(claims.amr)) redirect("/forgot-password?expired=1");

  const { error } = await supabase.auth.updateUser({ password: password as string });
  if (error) {
    if (error.code === "same_password") return { error: MESSAGES.samePassword };
    if (error.code === "weak_password") return { error: MESSAGES.weakPassword };
    if (isRateLimited(error)) return { error: MESSAGES.tooMany };
    return { error: MESSAGES.generic };
  }

  redirect(await homeFor(supabase, claims.sub));
}

export async function signOut(): Promise<void> {
  const supabase = await createClient();
  // Revokes the refresh token on Supabase's side and clears the cookies.
  await supabase.auth.signOut();
  redirect("/");
}
