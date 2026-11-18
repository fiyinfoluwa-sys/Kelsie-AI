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

/** POST { enabled } — the popup's Pause / Resume. */
export async function POST(request: Request) {
  if (!fromAllowedOrigin(request)) return forbidden();
  const token = bearerToken(request);
  if (!token) return unauthorized();

  let enabled: unknown;
  try {
    ({ enabled } = await request.json());
  } catch {
    return json({ error: "bad_request" }, 400);
  }
  if (typeof enabled !== "boolean") return json({ error: "bad_request" }, 400);

  const { data, error } = await anonClient().rpc("extension_set_memory", {
    p_token_hash: sha256(token),
    p_enabled: enabled,
  });
  if (error) {
    console.error("extension_set_memory failed:", error.code, error.message);
    return json({ error: "server_error" }, 500);
  }
  if (data !== true) return unauthorized();

  return json({ ok: true });
}

export function OPTIONS(request: Request) {
  return preflight(request);
}
