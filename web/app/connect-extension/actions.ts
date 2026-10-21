"use server";

import { newSecret, sha256 } from "@/lib/extension-auth";
import { requireUser } from "@/lib/session";

export type CodeResult = { ok: true; code: string } | { ok: false; error: string };

/**
 * Creates a single-use connection code for the signed-in user. Only the
 * code's hash is stored; the code itself goes straight to the extension
 * and expires after 5 minutes.
 */
export async function createExtensionCode(): Promise<CodeResult> {
  const { supabase } = await requireUser();
  const code = newSecret();
  const { error } = await supabase.rpc("create_extension_connect_code", { p_code_hash: sha256(code) });
  if (error) {
    return {
      ok: false,
      error:
        error.code === "54000"
          ? "Too many tries. Wait a few minutes and try again."
          : "Couldn't connect right now. Try again in a moment.",
    };
  }
  return { ok: true, code };
}
