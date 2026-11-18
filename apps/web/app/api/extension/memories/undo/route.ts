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
import { isUuid } from "@/lib/ids";

/** POST { id } — "Undo" in the orb: deletes a note the user just saved. */
export async function POST(request: Request) {
  if (!fromAllowedOrigin(request)) return forbidden();
  const token = bearerToken(request);
  if (!token) return unauthorized();

  let id: unknown;
  try {
    ({ id } = await request.json());
  } catch {
    return json({ error: "bad_request" }, 400);
  }
  if (!isUuid(id)) return json({ error: "bad_request" }, 400);

  const { data, error } = await anonClient().rpc("extension_delete_memory", {
    p_token_hash: sha256(token),
    p_memory_id: id,
  });
  if (error) {
    console.error("extension_delete_memory failed:", error.code);
    return json({ error: "server_error" }, 500);
  }
  return json({ deleted: data === true });
}

export function OPTIONS(request: Request) {
  return preflight(request);
}
