import type { SiteId, Vibe } from "@/lib/onboarding";

/*
 * Works out which setting a "settings_privacy" message is about, using
 * plain rules (no AI). Anything it can't pin down returns "unclear" and
 * goes to the AI for an explanation only: the AI never changes settings.
 */

export type SettingsRequest =
  | { kind: "delete_account" }
  | { kind: "clear_memories" }
  | { kind: "pause_memory" }
  | { kind: "resume_memory" }
  | { kind: "remove_site"; site: SiteId }
  | { kind: "add_site"; site: SiteId }
  | { kind: "unknown_site"; action: "remove" | "add" }
  | { kind: "set_vibe"; vibe: Vibe }
  | { kind: "unclear" };

const SITE_WORDS: [RegExp, SiteId][] = [
  [/\b(gmail|email|e-mail|inbox|mail)\b/, "gmail"],
  [/\blinked\s?in\b/, "linkedin"],
  [/\b(google docs?|docs|drive|google drive)\b/, "google_docs"],
  [/\b(canvas|school|uni|college|course|class) ?(sites?|portal)?\b/, "canvas"],
  [/\b(job boards?|indeed|glassdoor|greenhouse|lever|workday|wellfound|job sites?)\b/, "job_boards"],
  [/\b(every ?(other )?sites?|all (other )?sites|everywhere( else)?|the whole web|every ?website)\b/, "everything_else"],
];

const REMOVE = /\b(stop|remove|turn off|disable|don'?t (run|go|work)|no longer|block|exclude|get off|leave|quit)\b/;
const ADD = /\b(add|allow|start|turn on|enable|run on|work on|go on|let you)\b/;
const MEMORY = /\b(memor(y|ies)|remember(ing)?|notes?|saving|save stuff|what you (know|saved|remember))\b/;

export function parseSettingsRequest(text: string): SettingsRequest {
  const t = text.toLowerCase().replace(/[’']/g, "'");

  if (/\b(delete|close|deactivate|remove|erase|nuke|cancel)\b/.test(t) && /\b(account|profile|everything about me|all my data|my data)\b/.test(t)) {
    return { kind: "delete_account" };
  }
  if (/\b(clear|wipe|delete|erase|forget|remove|reset)\b/.test(t) && MEMORY.test(t)) {
    return { kind: "clear_memories" };
  }
  if (/\b(pause|stop|turn off|disable|don'?t)\b/.test(t) && MEMORY.test(t)) {
    return { kind: "pause_memory" };
  }
  if (/\b(resume|unpause|turn on|enable|start)\b/.test(t) && MEMORY.test(t)) {
    return { kind: "resume_memory" };
  }

  const vibe = /\b(vibe|tone|talk|sound|style)\b/.test(t)
    ? /\bhype\b/.test(t)
      ? "hype"
      : /\b(straight|direct|to the point|blunt)\b/.test(t)
        ? "direct"
        : /\bchill\b/.test(t)
          ? "chill"
          : null
    : null;
  if (vibe) return { kind: "set_vibe", vibe };

  const removing = REMOVE.test(t);
  const adding = !removing && ADD.test(t);
  if (removing || adding) {
    const site = SITE_WORDS.find(([pattern]) => pattern.test(t))?.[1];
    if (site) return removing ? { kind: "remove_site", site } : { kind: "add_site", site };
    // "stop running on reddit": a site that isn't one of Kelsie's site groups.
    if (/\b(on|from|for)\s+[a-z0-9.-]+/.test(t)) return { kind: "unknown_site", action: removing ? "remove" : "add" };
  }

  return { kind: "unclear" };
}
