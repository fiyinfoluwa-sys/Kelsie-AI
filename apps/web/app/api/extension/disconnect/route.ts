import {
  anonClient,
  bearerToken,
  forbidden,
  fromAllowedOrigin,
  json,
  preflight,
  sha256,
} from "@/lib/extension-auth";

/** POST — the popup's Disconnect. Deletes this extension's session. */
export async function POST(request: Request) {
  if (!fromAllowedOrigin(request)) return forbidden();
  const token = bearerToken(request);
  // Already disconnected as far as the extension needs to know.
  if (!token) return json({ ok: true });

  const { error } = await anonClient().rpc("extension_disconnect", { p_token_hash: sha256(token) });
  if (error) {
    console.error("extension_disconnect failed:", error.code, error.message);
    return json({ error: "server_error" }, 500);
  }
  return json({ ok: true });
}

export function OPTIONS(request: Request) {
  return preflight(request);
}
