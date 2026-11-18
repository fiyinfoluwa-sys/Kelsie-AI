import {
  anonClient,
  bearerToken,
  forbidden,
  fromAllowedOrigin,
  json,
  preflight,
  sha256,
  unauthorized,
} from "@/lib/extension-auth";

/**
 * GET → { name, vibe, allowed_sites, memory_enabled }
 * Everything the extension needs, for the token's own user only. A revoked
 * or unknown token gets 401, which makes the extension log itself out.
 */
export async function GET(request: Request) {
  if (!fromAllowedOrigin(request)) return forbidden();
  const token = bearerToken(request);
  if (!token) return unauthorized();

  const { data, error } = await anonClient().rpc("extension_state", { p_token_hash: sha256(token) });
  if (error) {
    console.error("extension_state failed:", error.code, error.message);
    return json({ error: "server_error" }, 500);
  }
  if (!data) return unauthorized();

  return json(data);
}

export function OPTIONS(request: Request) {
  return preflight(request);
}
