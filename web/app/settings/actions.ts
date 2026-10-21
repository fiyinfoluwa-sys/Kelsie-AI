"use server";

/*
 * Settings actions. Same rules as the profile actions: the user comes from
 * the verified session, every value is validated here, and Row Level
 * Security limits writes to the user's own rows.
 */

import { revalidatePath } from "next/cache";
import { isRateLimited, MESSAGES } from "@/lib/auth-messages";
import {
  checkName,
  checkRoles,
  checkSites,
  checkTimezone,
  checkVibe,
  sitesToList,
} from "@/lib/onboarding";
import { requireUser } from "@/lib/session";
import { siteOrigin } from "@/lib/site";

export type ActionResult = { ok: true } | { ok: false; error: string };

const TRY_AGAIN = "That didn't save. Try again in a moment.";

function refresh() {
  revalidatePath("/profile");
  revalidatePath("/settings");
}

/**
 * Saves one or more settings. Only these keys are accepted; anything else
 * (like an attempt to set another user's ID) is rejected.
 */
const SETTING_KEYS = new Set(["name", "roles", "timezone", "vibe", "sites", "memoryEnabled"]);

export async function updateSettings(changes: unknown): Promise<ActionResult> {
  if (typeof changes !== "object" || changes === null || Array.isArray(changes)) {
    return { ok: false, error: TRY_AGAIN };
  }
  const input = changes as Record<string, unknown>;
  const keys = Object.keys(input);
  if (keys.length === 0 || keys.some((k) => !SETTING_KEYS.has(k))) {
    return { ok: false, error: TRY_AGAIN };
  }

  const update: Record<string, unknown> = {};
  if ("name" in input) {
    const r = checkName(input.name);
    if (!r.ok) return r;
    update.name = r.value;
  }
  if ("roles" in input) {
    const r = checkRoles(input.roles);
    if (!r.ok) return r;
    update.roles = r.value;
  }
  if ("timezone" in input) {
    const r = checkTimezone(input.timezone);
    if (!r.ok) return r;
    update.timezone = r.value;
  }
  if ("vibe" in input) {
    const r = checkVibe(input.vibe);
    if (!r.ok) return r;
    update.vibe = r.value;
  }
  if ("sites" in input) {
    const r = checkSites(input.sites);
    if (!r.ok) return r;
    update.allowed_sites = sitesToList(r.value);
  }
  if ("memoryEnabled" in input) {
    if (typeof input.memoryEnabled !== "boolean") return { ok: false, error: TRY_AGAIN };
    update.memory_enabled = input.memoryEnabled;
  }

  const { supabase, user } = await requireUser();
  const { error } = await supabase.from("profiles").update(update).eq("id", user.id);
  if (error) return { ok: false, error: TRY_AGAIN };

  refresh();
  return { ok: true };
}

/** "Clear all memories": deletes every note Kelsie saved for this user. */
export async function clearMemories(): Promise<ActionResult> {
  const { supabase, user } = await requireUser();
  const { error } = await supabase.from("memories").delete().eq("user_id", user.id);
  if (error) return { ok: false, error: "Couldn't clear your memories. Try again in a moment." };

  refresh();
  return { ok: true };
}

/**
 * "Change password" for email accounts: emails a reset link to the
 * signed-in user's own address (taken from the session, not the request),
 * so changing the password always needs access to that inbox.
 */
export async function sendPasswordChangeEmail(): Promise<ActionResult> {
  const { supabase, user } = await requireUser();
  if (!user.email || user.app_metadata?.provider !== "email") {
    return { ok: false, error: "This account signs in with Google, so there's no password to change." };
  }
  const { error } = await supabase.auth.resetPasswordForEmail(user.email, {
    redirectTo: `${await siteOrigin()}/auth/confirm`,
  });
  if (error) {
    return { ok: false, error: isRateLimited(error) ? MESSAGES.tooMany : MESSAGES.generic };
  }
  return { ok: true };
}
