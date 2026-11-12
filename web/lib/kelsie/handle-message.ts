import "server-only";
import type { Routed } from "@/lib/intent-router";
import { redactPersonal } from "@/lib/memory-note";
import {
  handleApplication,
  handleDeadlineAdd,
  handleDeadlineQuery,
  handleFocus,
  handleGeneral,
  handleRecall,
  handleRemember,
  handleSettings,
  handleSmalltalk,
} from "./handlers";
import type { KelsieContext, KelsieReply } from "./types";

/** Sends a routed message to its handler. Anything the router wasn't sure about goes to the general AI. */
export function dispatch(routed: Routed, text: string, ctx: KelsieContext): Promise<KelsieReply> | KelsieReply {
  if (routed.fellBack) return handleGeneral(text, ctx);
  switch (routed.intent) {
    case "deadline_query":
      return handleDeadlineQuery(text, ctx);
    case "focus":
      return handleFocus(text, ctx);
    case "settings_privacy":
      return handleSettings(text, ctx);
    case "smalltalk":
      return handleSmalltalk(text, ctx);
    case "remember":
      return handleRemember(text, ctx);
    case "deadline_add":
      return handleDeadlineAdd(text, ctx);
    case "recall":
      return handleRecall(text, ctx);
    case "application":
      return handleApplication(text, ctx);
    default:
      return handleGeneral(text, ctx);
  }
}

/**
 * Logs what the router decided: intent, confidence, whether it fell back,
 * how long it took. Never the message.
 */
export async function logRoute(ctx: KelsieContext, routed: Routed, result: KelsieReply, totalMs: number) {
  const fields = {
    intent: routed.intent,
    predicted: routed.predicted,
    route: routed.routerError ? "router_error" : routed.route,
    handler: result.handler,
    confidence: routed.confidence,
    fellBack: routed.fellBack,
    usedAi: result.usedAi,
    routerMs: Math.round(routed.latencyMs * 100) / 100,
    totalMs: Math.round(totalMs),
  };
  console.log(JSON.stringify({ event: "kelsie_route", ...fields }));
  const { error } = await ctx.supabase.rpc("log_router_event", {
    p_intent: fields.intent,
    p_predicted: fields.predicted,
    p_route: fields.route,
    p_handler: fields.handler,
    p_confidence: fields.confidence,
    p_fell_back: fields.fellBack,
    p_used_ai: fields.usedAi,
    p_latency_ms: fields.routerMs,
  });
  if (error) console.error("log_router_event failed:", error.code);
}

/**
 * Only for testers who turned on "Help improve Kelsie": keeps the message
 * (with emails, numbers and links blanked out) and what the router guessed,
 * so it can be relabelled to retrain the model. The database refuses this
 * unless the setting is on.
 */
export async function saveTrainingExample(ctx: KelsieContext, routed: Routed, text: string) {
  const cleaned = redactPersonal(text)
    .replace(/https?:\/\/\S+|www\.\S+/gi, "[link]")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .trim()
    .slice(0, 500);
  if (!cleaned) return;
  const { error } = await ctx.supabase.from("router_training_examples").insert({
    user_id: ctx.userId,
    text: cleaned,
    predicted: routed.predicted,
    confidence: routed.confidence,
    fell_back: routed.fellBack,
  });
  if (error) console.error("saving training example failed:", error.code);
}
