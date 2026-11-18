import { routeMessage } from "@/lib/intent-router";
import { json } from "@/lib/kelsie/request";

/**
 * GET → whether the intent router loads on this deployment, using a fixed
 * test sentence. No user data in or out; useful right after a deploy.
 */
export async function GET() {
  const routed = await routeMessage("whats due this week");
  return json({
    ok: !routed.routerError && routed.intent === "deadline_query",
    intent: routed.intent,
    routerMs: Math.round(routed.latencyMs * 100) / 100,
  });
}
