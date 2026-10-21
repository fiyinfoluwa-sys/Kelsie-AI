import "server-only";
import { redirect } from "next/navigation";
import type { User } from "@supabase/supabase-js";
import { createClient } from "@/lib/supabase/server";
import type { ProfileRow } from "@/lib/onboarding";

type Supabase = Awaited<ReturnType<typeof createClient>>;

/**
 * The signed-in user, verified with Supabase Auth (getUser() asks the auth
 * server, so a deleted or banned account is caught immediately).
 * Redirects to sign-in when there's no verified, email-confirmed user.
 */
export async function requireUser(): Promise<{ supabase: Supabase; user: User }> {
  const supabase = await createClient();
  const { data, error } = await supabase.auth.getUser();
  const user = data?.user;
  if (error || !user) redirect("/signin");
  // Email sign-ups have no session until they verify, so this is a second
  // check in case the Supabase "Confirm email" setting is ever turned off.
  if (!user.email_confirmed_at) redirect("/signin?verify=1");
  return { supabase, user };
}

export async function getProfile(supabase: Supabase, userId: string): Promise<ProfileRow | null> {
  // Row Level Security already limits this to the user's own row; the
  // filter just makes the intent explicit.
  const { data, error } = await supabase
    .from("profiles")
    .select("name, roles, timezone, vibe, allowed_sites, onboarding_completed_at")
    .eq("id", userId)
    .maybeSingle();
  if (error) {
    // Details go to the server log only; the page shows a generic message.
    console.error("getProfile failed:", error.code, error.message, error.hint ?? "");
    throw new Error("Couldn't load your profile.");
  }
  return data;
}

/**
 * True when the session was created by an email link in the last hour.
 * Changing a password without the old one is only allowed then, so a
 * session left open on a shared computer can't be used to take over the
 * account.
 */
export function cameFromResetLink(amr: unknown): boolean {
  if (!Array.isArray(amr)) return false;
  const oneHourAgo = Date.now() / 1000 - 60 * 60;
  return amr.some(
    (entry) =>
      typeof entry === "object" &&
      entry !== null &&
      ["recovery", "otp"].includes((entry as { method?: string }).method ?? "") &&
      Number((entry as { timestamp?: number }).timestamp) > oneHourAgo,
  );
}

/** Where a signed-in user belongs: their profile, or onboarding if unfinished. */
export async function homeFor(supabase: Supabase, userId: string): Promise<string> {
  const profile = await getProfile(supabase, userId);
  return profile?.onboarding_completed_at ? "/profile" : "/onboarding";
}
