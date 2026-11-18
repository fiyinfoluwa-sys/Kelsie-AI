import "server-only";
import { createClient } from "@/lib/supabase/server";
import { displayName, rowToAnswers } from "@/lib/onboarding";
import { getProfile } from "@/lib/session";
import type { KelsieContext } from "./types";

/**
 * The signed-in user for an API request, or null (the route answers 401;
 * API routes don't redirect). Uses getUser(), which checks with Supabase
 * Auth, so a signed-out or deleted session can't get through.
 */
export async function loadContext(): Promise<(KelsieContext & { helpImprove: boolean }) | null> {
  const supabase = await createClient();
  const { data, error } = await supabase.auth.getUser();
  const user = data?.user;
  if (error || !user || !user.email_confirmed_at) return null;

  const profile = await getProfile(supabase, user.id);
  if (!profile?.onboarding_completed_at) return null;
  const answers = rowToAnswers(profile);

  const { data: extra } = await supabase.from("profiles").select("help_improve").eq("id", user.id).maybeSingle();
  return {
    supabase,
    userId: user.id,
    name: displayName(answers),
    vibe: answers.vibe,
    timezone: answers.timezone,
    memoryEnabled: profile.memory_enabled,
    allowedSites: Object.entries(answers.sites)
      .filter(([, on]) => on)
      .map(([id]) => id) as KelsieContext["allowedSites"],
    helpImprove: extra?.help_improve === true,
  };
}
