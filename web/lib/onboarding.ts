/*
 * Onboarding answers and where they're stored.
 *
 * Until real accounts exist, answers live in the browser's localStorage as a
 * single object. Everything else talks to this module, so moving to a
 * database later means changing loadAnswers/saveAnswers only.
 */

export type Role = "student" | "working" | "job_hunting" | "mix";
export type Vibe = "chill" | "hype" | "direct";
export type SiteId =
  | "google_docs"
  | "canvas"
  | "gmail"
  | "linkedin"
  | "job_boards"
  | "everything_else";

export type OnboardingAnswers = {
  version: 1;
  name: string;
  /** One or more of student/working/job_hunting, or just ["mix"]. */
  roles: Role[];
  timezone: string;
  vibe: Vibe | null;
  sites: Record<SiteId, boolean>;
  /** ISO timestamp, set when the user finishes setup. */
  completedAt: string | null;
};

export const ROLES: { id: Role; label: string; detail: string }[] = [
  { id: "student", label: "Student", detail: "Classes, assignments, exams" },
  { id: "working", label: "Working", detail: "Meetings, projects, emails" },
  { id: "job_hunting", label: "Job hunting", detail: "Applications and interviews" },
  { id: "mix", label: "A mix", detail: "A bit of everything" },
];

export const VIBES: { id: Vibe; label: string }[] = [
  { id: "chill", label: "Chill" },
  { id: "hype", label: "Hype" },
  { id: "direct", label: "Straight to the point" },
];

export const SITES: { id: SiteId; label: string }[] = [
  { id: "google_docs", label: "Google Docs and Drive" },
  { id: "canvas", label: "Canvas and school sites" },
  { id: "gmail", label: "Gmail" },
  { id: "linkedin", label: "LinkedIn" },
  { id: "job_boards", label: "Job boards" },
  { id: "everything_else", label: "Every other site" },
];

export function defaultAnswers(): OnboardingAnswers {
  return {
    version: 1,
    name: "",
    roles: [],
    timezone: "",
    vibe: null,
    sites: {
      google_docs: true,
      canvas: true,
      gmail: true,
      linkedin: true,
      job_boards: true,
      everything_else: false,
    },
    completedAt: null,
  };
}

const STORAGE_KEY = "kelsie.onboarding";

export function loadAnswers(): OnboardingAnswers {
  const defaults = defaultAnswers();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaults;
    const saved = JSON.parse(raw) as Partial<OnboardingAnswers> & {
      role?: Role | null;
    };
    if (saved.version !== 1) return defaults;
    // Early saves stored a single `role`; convert it to `roles`.
    const { role, ...rest } = saved;
    return {
      ...defaults,
      ...rest,
      roles: rest.roles ?? (role ? [role] : []),
      sites: { ...defaults.sites, ...saved.sites },
    };
  } catch {
    // Storage can be blocked (private mode) or hold bad data.
    return defaults;
  }
}

export function saveAnswers(answers: OnboardingAnswers): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
  } catch {
    // Not fatal: the flow still works for this visit.
  }
}

/**
 * Toggle a role. "A mix" can't be combined with anything: picking it
 * clears the others, and picking any other role clears it.
 */
export function toggleRole(roles: Role[], role: Role): Role[] {
  if (roles.includes(role)) return roles.filter((r) => r !== role);
  if (role === "mix") return ["mix"];
  return [...roles.filter((r) => r !== "mix"), role];
}

/** The name Kelsie uses in messages. */
export function displayName(answers: Pick<OnboardingAnswers, "name">): string {
  return answers.name.trim() || "friend";
}

export function vibeSample(vibe: Vibe, name: string): string {
  switch (vibe) {
    case "chill":
      return `hey ${name}, no rush, but your econ essay is due thursday. want your notes?`;
    case "hype":
      return `${name}!! econ essay due thursday and you're SO ready for it. want me to pull up your notes?`;
    case "direct":
      return `${name}: econ essay due Thursday. Your notes are ready.`;
  }
}

export function vibeFarewell(vibe: Vibe | null, name: string): string {
  switch (vibe) {
    case "hype":
      return `${name}, LET'S GOOO. this is gonna be a great semester.`;
    case "direct":
      return `Setup complete, ${name}. Add the extension to start.`;
    case "chill":
    default:
      return `all set, ${name}. I'll be around when you need me.`;
  }
}
