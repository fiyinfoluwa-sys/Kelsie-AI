"use server";

/*
 * Profile page actions. Each one gets the user from the verified session
 * (never from the request), validates its input, and only touches that
 * user's rows. Row Level Security enforces the same rule in the database.
 */

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { DELETE_CONFIRMATION } from "@/lib/account";
import { MEMORY_MAX_LENGTH } from "@/lib/content";
import { isUuid } from "@/lib/ids";
import { SITES, checkSiteId, type SiteId } from "@/lib/onboarding";
import { requireUser } from "@/lib/session";

export type ActionResult = { ok: true } | { ok: false; error: string };
export type DeleteState = { error?: string };

const TRY_AGAIN = "That didn't save. Try again in a moment.";

function refresh() {
  revalidatePath("/profile");
  revalidatePath("/settings");
}

// ------------------------------------------------------------
// Memories
// ------------------------------------------------------------

function checkMemoryText(input: unknown): { ok: true; value: string } | { ok: false; error: string } {
  if (typeof input !== "string") return { ok: false, error: "Write something first." };
  const text = input.replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, "").trim();
  if (!text) return { ok: false, error: "Write something first, or delete the note." };
  if (text.length > MEMORY_MAX_LENGTH) {
    return { ok: false, error: `Notes can be up to ${MEMORY_MAX_LENGTH} characters.` };
  }
  return { ok: true, value: text };
}

export async function updateMemory(id: unknown, content: unknown): Promise<ActionResult> {
  if (!isUuid(id)) return { ok: false, error: TRY_AGAIN };
  const text = checkMemoryText(content);
  if (!text.ok) return text;

  const { supabase, user } = await requireUser();
  const { data, error } = await supabase
    .from("memories")
    .update({ content: text.value })
    .eq("id", id)
    .eq("user_id", user.id)
    .select("id");
  if (error || !data?.length) return { ok: false, error: TRY_AGAIN };

  refresh();
  return { ok: true };
}

export async function deleteMemory(id: unknown): Promise<ActionResult> {
  if (!isUuid(id)) return { ok: false, error: TRY_AGAIN };
  const { supabase, user } = await requireUser();
  const { error } = await supabase.from("memories").delete().eq("id", id).eq("user_id", user.id);
  if (error) return { ok: false, error: TRY_AGAIN };

  refresh();
  return { ok: true };
}

// ------------------------------------------------------------
// Deadlines
// ------------------------------------------------------------

/** "Add it" (confirm) or "Not a deadline" (dismiss) for a suggested deadline. */
export async function reviewDeadline(id: unknown, decision: unknown): Promise<ActionResult> {
  if (!isUuid(id) || (decision !== "confirmed" && decision !== "dismissed")) {
    return { ok: false, error: TRY_AGAIN };
  }
  const { supabase, user } = await requireUser();
  const { data, error } = await supabase
    .from("deadlines")
    .update({ status: decision })
    .eq("id", id)
    .eq("user_id", user.id)
    .eq("status", "suggested")
    .select("id");
  if (error || !data?.length) return { ok: false, error: TRY_AGAIN };

  refresh();
  return { ok: true };
}

// ------------------------------------------------------------
// Sites (shared with the Settings checkboxes: both edit allowed_sites)
// ------------------------------------------------------------

export type SitesResult = { ok: true; sites: SiteId[] } | { ok: false; error: string };

/** Turns one site on or off and returns the updated list. */
export async function setSiteAllowed(site: unknown, allowed: unknown): Promise<SitesResult> {
  const checked = checkSiteId(site);
  if (!checked.ok || typeof allowed !== "boolean") return { ok: false, error: TRY_AGAIN };

  const { supabase, user } = await requireUser();
  const { data: profile, error: readError } = await supabase
    .from("profiles")
    .select("allowed_sites")
    .eq("id", user.id)
    .single();
  if (readError || !profile) return { ok: false, error: TRY_AGAIN };

  const current = new Set<string>(profile.allowed_sites);
  if (allowed) current.add(checked.value);
  else current.delete(checked.value);
  // Keep the usual order so the list doesn't shuffle.
  const sites = SITES.map((s) => s.id).filter((id) => current.has(id));

  const { error } = await supabase.from("profiles").update({ allowed_sites: sites }).eq("id", user.id);
  if (error) return { ok: false, error: TRY_AGAIN };

  refresh();
  return { ok: true, sites };
}

// ------------------------------------------------------------
// Account deletion
// ------------------------------------------------------------

/**
 * Deletes the signed-in user's account and everything stored for them.
 * The delete_my_account() database function can only delete the caller's
 * own account, so the app doesn't need the service role key.
 */
export async function deleteAccount(_prev: DeleteState, form: FormData): Promise<DeleteState> {
  if (form.get("confirm") !== DELETE_CONFIRMATION) {
    return { error: `Type ${DELETE_CONFIRMATION} to confirm.` };
  }

  const { supabase } = await requireUser();
  const { error } = await supabase.rpc("delete_my_account");
  if (error) return { error: "Couldn't delete your account. Try again in a moment." };

  // The account is gone; clear this browser's session cookies too.
  await supabase.auth.signOut({ scope: "local" });
  redirect("/?deleted=1");
}
