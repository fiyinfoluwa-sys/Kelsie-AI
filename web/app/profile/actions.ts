"use server";

import { redirect } from "next/navigation";
import { DELETE_CONFIRMATION } from "@/lib/account";
import { requireUser } from "@/lib/session";

export type DeleteState = { error?: string };

/**
 * Deletes the signed-in user's account and everything stored for them.
 *
 * Calls the delete_my_account() database function, which can only ever
 * delete the caller's own account (it uses the verified user ID from the
 * session token), so the app doesn't need the service role key.
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
