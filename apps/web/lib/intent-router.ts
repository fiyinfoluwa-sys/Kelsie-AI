import "server-only";
import path from "node:path";
import { createRouter, type RouteResult, type Router } from "@/intent-router/router.js";

/*
 * Kelsie's intent router (trained in /kelsie-intent, copied into
 * /intent-router). Loaded once per server instance, the first time a
 * message needs it, then reused for every request.
 */

let loading: Promise<Router> | null = null;

function loadRouter(): Promise<Router> {
  if (!loading) {
    const dir = path.join(process.cwd(), "intent-router");
    loading = createRouter(path.join(dir, "intent_int8.onnx"), path.join(dir, "router_config.json")).catch(
      (error: unknown) => {
        // Let the next request try again instead of caching the failure.
        loading = null;
        throw error;
      },
    );
  }
  return loading;
}

export type Routed = RouteResult & { latencyMs: number; routerError: boolean };

/**
 * Classifies a message. If the model can't load, it falls back to the
 * general LLM route rather than failing the request.
 */
export async function routeMessage(text: string): Promise<Routed> {
  const started = performance.now();
  try {
    const router = await loadRouter();
    const result = await router.route(text);
    return { ...result, latencyMs: performance.now() - started, routerError: false };
  } catch {
    console.error("Intent router unavailable; using the general route");
    return {
      text,
      predicted: "general",
      confidence: 0,
      intent: "general",
      route: "llm",
      fellBack: true,
      latencyMs: performance.now() - started,
      routerError: true,
    };
  }
}
