import type { Vibe } from "@/lib/onboarding";

/*
 * Kelsie's canned lines, in each vibe. Used for everything the router
 * handles without an AI call.
 */

type Lines = Record<Vibe, string>;

/** Picks the line for the user's vibe (chill if they haven't picked one). */
export function say(vibe: Vibe | null, lines: Lines): string {
  return lines[vibe ?? "chill"];
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function deadlinesLine(vibe: Vibe | null, when: string, items: string[]): string {
  if (items.length === 0) {
    return say(vibe, {
      chill: `nothing due ${when}. enjoy it.`,
      hype: `NOTHING due ${when}. love that for you!`,
      direct: `Nothing due ${when}.`,
    });
  }
  const list = items.join(", ");
  return say(vibe, {
    chill: `you've got ${plural(items.length, "thing")} due ${when}: ${list}. no rush, one at a time.`,
    hype: `${plural(items.length, "thing")} due ${when} and you're SO ready: ${list}!`,
    direct: `${items.length} due ${when}: ${list}.`,
  });
}

export function focusStarted(vibe: Vibe | null, minutes: number): string {
  return say(vibe, {
    chill: `locked in for ${minutes} min. I'll keep things quiet till then.`,
    hype: `LET'S LOCK IN. ${minutes} minutes, starting now!`,
    direct: `Focus on: ${minutes} min.`,
  });
}

export function focusAlreadyOn(vibe: Vibe | null, minutesLeft: number): string {
  return say(vibe, {
    chill: `you're already locked in, ${plural(minutesLeft, "minute")} to go.`,
    hype: `already locked in! ${plural(minutesLeft, "minute")} left, keep going!`,
    direct: `Already focusing. ${minutesLeft} min left.`,
  });
}

export function focusExtended(vibe: Vibe | null, added: number, minutesLeft: number): string {
  return say(vibe, {
    chill: `added ${added} min. ${minutesLeft} to go now.`,
    hype: `+${added} minutes! ${minutesLeft} left, you've got this!`,
    direct: `Extended by ${added} min. ${minutesLeft} min left.`,
  });
}

export function focusStopped(vibe: Vibe | null, minutesDone: number): string {
  return say(vibe, {
    chill: `nice, ${plural(minutesDone, "minute")} of focus. take a breather.`,
    hype: `${plural(minutesDone, "minute")} LOCKED IN. proud of you!`,
    direct: `Focus ended after ${minutesDone} min.`,
  });
}

export function focusNotRunning(vibe: Vibe | null): string {
  return say(vibe, {
    chill: "you're not in a focus session right now. want to start one?",
    hype: "no focus session running! say the word and we lock in.",
    direct: "No focus session running.",
  });
}

export function noteSaved(vibe: Vibe | null, note: string): string {
  return say(vibe, {
    chill: `saved: "${note}"`,
    hype: `got it! saved: "${note}"`,
    direct: `Saved: "${note}"`,
  });
}

export function memoryPaused(vibe: Vibe | null): string {
  return say(vibe, {
    chill: "memory's paused right now, so I didn't save that. you can resume it in Settings.",
    hype: "memory's paused so I couldn't save that! turn it back on in Settings and I'm all yours.",
    direct: "Memory is paused, so nothing was saved. Resume it in Settings.",
  });
}

export function aiLimitReached(vibe: Vibe | null): string {
  return say(vibe, {
    chill: "I've hit my thinking limit for today. try again tomorrow?",
    hype: "I'm all out of brainpower for today! back tomorrow, promise.",
    direct: "Daily limit reached. Try again tomorrow.",
  });
}

export function aiUnavailable(vibe: Vibe | null): string {
  return say(vibe, {
    chill: "I can't think that one through right now. try again in a minute?",
    hype: "ugh, my brain's busy right now! try me again in a sec?",
    direct: "Can't answer that right now. Try again shortly.",
  });
}

/** Smalltalk: greetings, thanks, goodbyes. */
export function smalltalk(vibe: Vibe | null, name: string, text: string): string {
  const t = text.toLowerCase();
  if (/\b(thanks|thank you|thx|ty|tysm|appreciate)\b/.test(t)) {
    return say(vibe, { chill: `anytime, ${name}.`, hype: `ANYTIME, ${name}!!`, direct: "You're welcome." });
  }
  if (/\b(bye|goodnight|good night|gn|later|cya|see you)\b/.test(t)) {
    return say(vibe, { chill: `later, ${name}.`, hype: `byeee ${name}! go crush it.`, direct: "Bye." });
  }
  if (/\b(how are you|how r u|hows it going|how's it going|wyd|what's up|whats up|sup)\b/.test(t)) {
    return say(vibe, {
      chill: "all good, just riding along. what's up with you?",
      hype: "THRIVING. what are we getting done today?",
      direct: "Good. What do you need?",
    });
  }
  if (/\b(hi|hey|hello|yo|hiya|morning|evening)\b/.test(t)) {
    return say(vibe, {
      chill: `hey ${name}. what's up?`,
      hype: `${name}!! hiii. what are we doing today?`,
      direct: `Hi ${name}. What do you need?`,
    });
  }
  return say(vibe, {
    chill: "I'm here whenever you need me.",
    hype: "love that! I'm right here if you need me.",
    direct: "Noted.",
  });
}
