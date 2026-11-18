/*
 * Display text shared by the profile and settings pages.
 */
import type { SiteId, Vibe } from "./onboarding";

export const MEMORY_MAX_LENGTH = 500;

export type MemorySource = SiteId | "other";
export type ApplicationStatus = "draft" | "applied" | "interview" | "offer" | "rejected";

export const SOURCE_LABELS: Record<MemorySource, string> = {
  google_docs: "Google Docs",
  canvas: "Canvas",
  gmail: "Gmail",
  linkedin: "LinkedIn",
  job_boards: "Job board",
  everything_else: "The web",
  other: "Kelsie",
};

/** What Kelsie does on each site, shown in "Where Kelsie's riding along". */
export const SITE_DESCRIPTIONS: Record<SiteId, string> = {
  google_docs: "Pulls quotes and notes into what you're writing.",
  canvas: "Spots due dates on assignments and syllabi.",
  gmail: "Catches deadlines and follow-ups in your inbox.",
  linkedin: "Helps with job posts and keeps track of recruiters.",
  job_boards: "Autofills applications and tracks where you applied.",
  everything_else: "Notices useful moments anywhere else you browse.",
};

export const APPLICATION_STATUS_LABELS: Record<ApplicationStatus, string> = {
  draft: "Draft saved",
  applied: "Applied",
  interview: "Interview",
  offer: "Offer",
  rejected: "Closed",
};

function partOfDay(hour: number): "morning" | "afternoon" | "evening" {
  if (hour < 12) return "morning";
  if (hour < 17) return "afternoon";
  return "evening";
}

/** The greeting under the orb, in the user's vibe. */
export function greeting(
  vibe: Vibe | null,
  name: string,
  hour: number,
  upcoming: number,
): { title: string; line: string } {
  const part = partOfDay(hour);
  const plural = upcoming === 1 ? "" : "s";
  switch (vibe) {
    case "hype":
      return {
        title: `GOOD ${part.toUpperCase()}, ${name}!`,
        line: upcoming
          ? `${upcoming} thing${plural} on deck and you're SO ready.`
          : "clear schedule? love that for you.",
      };
    case "direct":
      return {
        title: `Good ${part}, ${name}.`,
        line: upcoming ? `${upcoming} deadline${plural} coming up.` : "Nothing due.",
      };
    case "chill":
    default:
      return {
        title: `good ${part}, ${name}.`,
        line: upcoming
          ? `${upcoming} thing${plural} coming up. no rush, I've got the list.`
          : "nothing due soon. enjoy it.",
      };
  }
}

/** Applications appear for job hunters (or "a mix") who have some saved. */
export function showsApplications(roles: string[], count: number): boolean {
  return count > 0 && (roles.includes("job_hunting") || roles.includes("mix"));
}
