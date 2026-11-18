import "server-only";
import type { createClient } from "@/lib/supabase/server";
import type { ApplicationStatus, MemorySource } from "./content";
import { localDate } from "./dates";

type Supabase = Awaited<ReturnType<typeof createClient>>;

export type Memory = { id: string; content: string; source: MemorySource; created_at: string };
export type Deadline = {
  id: string;
  title: string;
  details: string;
  due_date: string;
  status: "suggested" | "confirmed" | "dismissed";
};
export type Application = {
  id: string;
  company: string;
  role: string;
  status: ApplicationStatus;
  applied_on: string | null;
};

/**
 * Everything the profile page shows. Row Level Security already limits
 * each query to the signed-in user's rows; the user_id filters make that
 * explicit.
 */
export async function loadDashboard(supabase: Supabase, userId: string, timezone: string) {
  const today = localDate(timezone);

  const [memories, upcoming, suggested, applications] = await Promise.all([
    supabase
      .from("memories")
      .select("id, content, source, created_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(20),
    supabase
      .from("deadlines")
      .select("id, title, details, due_date, status")
      .eq("user_id", userId)
      .eq("status", "confirmed")
      .gte("due_date", today)
      .order("due_date")
      .limit(6),
    supabase
      .from("deadlines")
      .select("id, title, details, due_date, status")
      .eq("user_id", userId)
      .eq("status", "suggested")
      .gte("due_date", today)
      .order("due_date")
      .limit(3),
    supabase
      .from("applications")
      .select("id, company, role, status, applied_on")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(50),
  ]);

  for (const result of [memories, upcoming, suggested, applications]) {
    if (result.error) {
      console.error("loadDashboard failed:", result.error.code, result.error.message);
      throw new Error("Couldn't load your profile.");
    }
  }

  return {
    today,
    memories: (memories.data ?? []) as Memory[],
    upcoming: (upcoming.data ?? []) as Deadline[],
    suggested: (suggested.data ?? []) as Deadline[],
    applications: (applications.data ?? []) as Application[],
  };
}
