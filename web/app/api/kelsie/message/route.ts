import { routeMessage } from "@/lib/intent-router";
import { loadContext } from "@/lib/kelsie/context";
import { dispatch, logRoute, saveTrainingExample } from "@/lib/kelsie/handle-message";
import { isSameOrigin, json } from "@/lib/kelsie/request";

const MAX_MESSAGE_CHARS = 1000;

/**
 * POST { text } → { reply, handler, intent, confirm? }
 *
 * For the signed-in user's own messages to Kelsie (session cookie). The
 * intent router picks a handler before any AI call; see lib/kelsie.
 * Cross-site requests can't use this: the session cookie is SameSite=Lax,
 * so it isn't sent on another site's POST.
 */
export async function POST(request: Request) {
  const started = performance.now();
  if (!isSameOrigin(request)) return json({ error: "forbidden" }, 403);
  const ctx = await loadContext();
  if (!ctx) return json({ error: "not_signed_in" }, 401);

  let text: unknown;
  try {
    ({ text } = await request.json());
  } catch {
    return json({ error: "bad_request" }, 400);
  }
  if (typeof text !== "string") return json({ error: "bad_request" }, 400);
  const message = text.replace(/[\u0000-\u0008\u000b-\u001f\u007f]/g, " ").trim();
  if (!message) return json({ error: "empty" }, 400);
  if (message.length > MAX_MESSAGE_CHARS) return json({ error: "too_long" }, 413);

  const routed = await routeMessage(message);
  const result = await dispatch(routed, message, ctx);

  await logRoute(ctx, routed, result, performance.now() - started);
  if (ctx.helpImprove) await saveTrainingExample(ctx, routed, message);

  return json({
    reply: result.reply,
    handler: result.handler,
    intent: routed.intent,
    ...(result.confirm ? { confirm: result.confirm } : {}),
  });
}
