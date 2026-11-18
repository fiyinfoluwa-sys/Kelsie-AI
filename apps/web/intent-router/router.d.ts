// Types for router.js (copied unchanged from kelsie-intent/js/router.js).

export type Intent =
  | "remember"
  | "recall"
  | "deadline_add"
  | "deadline_query"
  | "focus"
  | "application"
  | "settings_privacy"
  | "smalltalk"
  | "general";

export type Route = "local" | "llm_small" | "llm";

export type RouteResult = {
  text: string;
  predicted: Intent;
  confidence: number;
  intent: Intent;
  route: Route;
  fellBack: boolean;
};

export type Router = {
  route(text: string): Promise<RouteResult>;
  probs(texts: string[]): Promise<number[][]>;
  config: {
    labels: Intent[];
    routes: Record<Intent, Route>;
    fallback: Intent;
    temperature: number;
    threshold: number;
  };
};

export function createRouter(modelPath: string, configPath: string): Promise<Router>;
