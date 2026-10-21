import "server-only";

/*
 * The one place Kelsie talks to an AI model. To switch providers, change
 * this file only.
 *
 * Provider: Groq (OpenAI-compatible API). Groq doesn't keep request data
 * by default; turning on Zero Data Retention in Groq's Data Controls also
 * stops the short-term abuse-monitoring logs.
 *
 * Privacy: prompts and replies are never logged here. Errors log only the
 * HTTP status.
 */

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const DEFAULT_MODEL = "llama-3.1-8b-instant";
const TIMEOUT_MS = 12_000;

export type AiResult =
  | { ok: true; text: string }
  | { ok: false; reason: "not_configured" | "rate_limited" | "error" };

/**
 * Asks the model for a JSON object. Returns the raw JSON text; callers
 * must validate it. Never retries: if the provider is busy or over its
 * limit, the caller just skips.
 */
export async function completeJson(options: {
  system: string;
  user: string;
  maxTokens: number;
}): Promise<AiResult> {
  const key = process.env.GROQ_API_KEY;
  if (!key) return { ok: false, reason: "not_configured" };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(GROQ_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: process.env.GROQ_MODEL || DEFAULT_MODEL,
        messages: [
          { role: "system", content: options.system },
          { role: "user", content: options.user },
        ],
        temperature: 0.2,
        max_tokens: options.maxTokens,
        response_format: { type: "json_object" },
      }),
      signal: controller.signal,
      cache: "no-store",
    });

    if (response.status === 429) return { ok: false, reason: "rate_limited" };
    if (!response.ok) {
      console.error("AI request failed with status", response.status);
      return { ok: false, reason: "error" };
    }
    const data = (await response.json()) as { choices?: { message?: { content?: unknown } }[] };
    const text = data.choices?.[0]?.message?.content;
    return typeof text === "string" ? { ok: true, text } : { ok: false, reason: "error" };
  } catch {
    // Timeout or network error. No details logged (they could echo input).
    console.error("AI request failed (timeout or network)");
    return { ok: false, reason: "error" };
  } finally {
    clearTimeout(timer);
  }
}
