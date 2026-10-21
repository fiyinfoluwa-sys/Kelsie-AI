"use server";

import { sitesToList, validateAnswers } from "@/lib/onboarding";
import { requireUser } from "@/lib/session";

export type SaveResult = { ok: true } | { ok: false; error: string };

/**
 * Saves onboarding answers for the signed-in user.
 *
 * `input` comes from the browser, so it's treated as untrusted: it's
 * validated here, and the user ID comes from the verified session, never
 * from the request. Row Level Security would block writing anyone else's
 * row anyway.
 */
export async function saveOnboarding(input: unknown): Promise<SaveResult> {
  const { supabase, user } = await requireUser();

  const result = validateAnswers(input);
  if (!result.ok) return result;
  const answers = result.value;

  const { error } = await supabase.from("profiles").upsert({
    id: user.id,
    name: answers.name,
    roles: answers.roles,
    timezone: answers.timezone,
    vibe: answers.vibe,
    allowed_sites: sitesToList(answers.sites),
    onboarding_completed_at: new Date().toISOString(),
  });
  if (error) return { ok: false, error: "Couldn't save your answers. Try again in a moment." };

  return { ok: true };
}
