import type { createClient } from "@/lib/supabase/server";
import type { SiteId, Vibe } from "@/lib/onboarding";

export type Supabase = Awaited<ReturnType<typeof createClient>>;

/** Who's talking to Kelsie, loaded from their own profile. */
export type KelsieContext = {
  supabase: Supabase;
  userId: string;
  name: string;
  vibe: Vibe | null;
  timezone: string;
  memoryEnabled: boolean;
  allowedSites: SiteId[];
};

/**
 * Something Kelsie wants to do but needs a yes from the user first. The
 * route never does these itself; the UI asks, then calls /api/kelsie/confirm
 * (or, for deleting the account, sends the user to Settings).
 */
export type ConfirmStep =
  | { action: "delete_account" }
  | { action: "clear_memories" }
  | { action: "remove_site"; site: SiteId }
  | { action: "add_site"; site: SiteId }
  | { action: "add_deadline"; title: string; dueDate: string };

export type Handler =
  | "deadline_query"
  | "focus"
  | "settings"
  | "smalltalk"
  | "remember"
  | "deadline_add"
  | "recall"
  | "application"
  | "general";

export type KelsieReply = {
  handler: Handler;
  reply: string;
  confirm?: ConfirmStep;
  /** Whether an AI call was made (for counting what the router saves). */
  usedAi: boolean;
};
