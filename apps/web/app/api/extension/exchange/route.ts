import {
  anonClient,
  forbidden,
  fromAllowedOrigin,
  isSecretShape,
  json,
  newSecret,
  preflight,
  sha256,
} from "@/lib/extension-auth";

/**
 * POST { code } → { token }
 * Trades a single-use connection code (made on /connect-extension) for an
 * extension session token. The plain token is returned once, here, and
 * only its hash is stored.
 */
export async function POST(request: Request) {
  if (!fromAllowedOrigin(request)) return forbidden();
  let code: unknown;
  try {
    ({ code } = await request.json());
  } catch {
    return json({ error: "bad_request" }, 400);
  }
  if (!isSecretShape(code)) return json({ error: "bad_request" }, 400);

  const token = newSecret();
  const { data, error } = await anonClient().rpc("exchange_extension_code", {
    p_code_hash: sha256(code),
    p_token_hash: sha256(token),
  });
  if (error) {
    console.error("exchange_extension_code failed:", error.code, error.message);
    return json({ error: "server_error" }, 500);
  }
  if (data !== true) return json({ error: "invalid_code" }, 400);

  return json({ token });
}

export function OPTIONS(request: Request) {
  return preflight(request);
}
