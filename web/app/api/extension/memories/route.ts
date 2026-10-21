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
import { MIN_TEXT_LENGTH, preparePageText, writeMemoryNote } from "@/lib/memory-note";
import { SITES, checkSiteId } from "@/lib/onboarding";

// Generous ceiling for { title, text } after the extension's own 6,000
// character cap; anything bigger is refused before it's read.
const MAX_BODY_BYTES = 40_000;

/**
 * POST { source, pageKey, title, text } → { saved, id? , reason? }
 *
 * Writes one short note about a page and saves only the note. The page
 * text lives in this request's memory and is dropped when it ends: it's
 * never stored, and never logged. Every "can't save" case answers 200
 * with saved: false so the extension just quietly skips.
 */
export async function POST(request: Request) {
  if (!fromAllowedOrigin(request)) return forbidden();
  const token = bearerToken(request);
  if (!token) return unauthorized();

  const length = Number(request.headers.get("content-length") ?? "0");
  if (length > MAX_BODY_BYTES) return json({ error: "too_large" }, 413);

  let raw: string;
  let body: Record<string, unknown>;
  try {
    raw = await request.text();
    if (raw.length > MAX_BODY_BYTES) return json({ error: "too_large" }, 413);
    body = JSON.parse(raw);
  } catch {
    return json({ error: "bad_request" }, 400);
  }

  const source = checkSiteId(body.source);
  const pageKey = typeof body.pageKey === "string" && /^[0-9a-f]{64}$/.test(body.pageKey) ? body.pageKey : null;
  if (!source.ok || !pageKey || typeof body.title !== "string" || typeof body.text !== "string") {
    return json({ error: "bad_request" }, 400);
  }
  const title = body.title.replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim().slice(0, 200);
  const text = preparePageText(body.text);
  if (text.length < MIN_TEXT_LENGTH) return json({ saved: false, reason: "too_short" });

  const db = anonClient();
  const tokenHash = sha256(token);

  // Checks session, pause, site and limits, and counts the attempt.
  const { data: reservation, error: reserveError } = await db.rpc("extension_reserve_memory", {
    p_token_hash: tokenHash,
    p_source: source.value,
  });
  if (reserveError) {
    console.error("extension_reserve_memory failed:", reserveError.code);
    return json({ saved: false, reason: "server_error" });
  }
  if (reservation === "not_connected") return unauthorized();
  if (reservation !== "ok") return json({ saved: false, reason: reservation });

  const label = SITES.find((s) => s.id === source.value)?.label ?? "the web";
  const note = await writeMemoryNote(label, title, text);
  if (!note.ok) return json({ saved: false, reason: note.reason === "rate_limited" ? "busy" : "ai_unavailable" });

  const { data: id, error: saveError } = await db.rpc("extension_save_memory", {
    p_token_hash: tokenHash,
    p_page_key: pageKey,
    p_source: source.value,
    p_title: title,
    p_note: note.note,
  });
  if (saveError) {
    console.error("extension_save_memory failed:", saveError.code);
    return json({ saved: false, reason: "server_error" });
  }
  if (!id) return json({ saved: false, reason: "not_allowed" });

  return json({ saved: true, id });
}

export function OPTIONS(request: Request) {
  return preflight(request);
}
