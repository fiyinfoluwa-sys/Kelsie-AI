/*
 * Onboarding answers: types, allowed values, validation and Kelsie's lines.
 *
 * Shared by the onboarding screen (browser) and the save action (server).
 * The server always re-validates with validateAnswers() before writing, and
 * the database has matching CHECK constraints as a final backstop.
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
  name: string;
  /** One or more of student/working/job_hunting, or just ["mix"]. */
  roles: Role[];
  timezone: string;
  vibe: Vibe | null;
  sites: Record<SiteId, boolean>;
};

export const NAME_MAX_LENGTH = 40;

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

export const DEFAULT_SITES: Record<SiteId, boolean> = {
  google_docs: true,
  canvas: true,
  gmail: true,
  linkedin: true,
  job_boards: true,
  everything_else: false,
};

export function defaultAnswers(): OnboardingAnswers {
  return { name: "", roles: [], timezone: "", vibe: null, sites: { ...DEFAULT_SITES } };
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

// ------------------------------------------------------------
// Validation
// ------------------------------------------------------------

const ROLE_IDS = new Set<string>(ROLES.map((r) => r.id));
const VIBE_IDS = new Set<string>(VIBES.map((v) => v.id));
const SITE_IDS = SITES.map((s) => s.id);

export type ValidationResult =
  | { ok: true; value: OnboardingAnswers }
  | { ok: false; error: string };

function isValidTimezone(zone: string): boolean {
  if (zone.length < 1 || zone.length > 64) return false;
  if (!/^[A-Za-z0-9_+-]+(\/[A-Za-z0-9_+-]+)*$/.test(zone)) return false;
  try {
    // Throws a RangeError for anything that isn't a real IANA zone.
    new Intl.DateTimeFormat("en-US", { timeZone: zone });
    return true;
  } catch {
    return false;
  }
}

/**
 * Checks untrusted input (it arrives from the browser) and returns a clean
 * copy containing only known fields with allowed values.
 */
export function validateAnswers(input: unknown): ValidationResult {
  if (typeof input !== "object" || input === null) {
    return { ok: false, error: "Something went wrong. Please try again." };
  }
  const raw = input as Record<string, unknown>;

  if (typeof raw.name !== "string") return { ok: false, error: "Please enter a name." };
  // Collapse whitespace and drop control characters.
  const name = raw.name.replace(/[\u0000-\u001f\u007f]/g, "").replace(/\s+/g, " ").trim();
  if (name.length > NAME_MAX_LENGTH) {
    return { ok: false, error: `Names can be up to ${NAME_MAX_LENGTH} characters.` };
  }

  if (
    !Array.isArray(raw.roles) ||
    raw.roles.length === 0 ||
    raw.roles.length > 3 ||
    !raw.roles.every((r) => typeof r === "string" && ROLE_IDS.has(r)) ||
    new Set(raw.roles).size !== raw.roles.length ||
    (raw.roles.includes("mix") && raw.roles.length > 1)
  ) {
    return { ok: false, error: "Pick what you're up to." };
  }
  const roles = raw.roles as Role[];

  if (typeof raw.timezone !== "string" || !isValidTimezone(raw.timezone)) {
    return { ok: false, error: "Pick a valid timezone." };
  }

  if (typeof raw.vibe !== "string" || !VIBE_IDS.has(raw.vibe)) {
    return { ok: false, error: "Pick a vibe for Kelsie." };
  }
  const vibe = raw.vibe as Vibe;

  if (typeof raw.sites !== "object" || raw.sites === null) {
    return { ok: false, error: "Something went wrong with your sites. Please try again." };
  }
  const rawSites = raw.sites as Record<string, unknown>;
  const sites = { ...DEFAULT_SITES };
  for (const id of SITE_IDS) {
    if (typeof rawSites[id] !== "boolean") {
      return { ok: false, error: "Something went wrong with your sites. Please try again." };
    }
    sites[id] = rawSites[id] as boolean;
  }

  return { ok: true, value: { name, roles, timezone: raw.timezone, vibe, sites } };
}

// ------------------------------------------------------------
// Database row mapping
// ------------------------------------------------------------

export type ProfileRow = {
  name: string;
  roles: string[];
  timezone: string;
  vibe: string | null;
  allowed_sites: string[];
  onboarding_completed_at: string | null;
};

export function sitesToList(sites: Record<SiteId, boolean>): SiteId[] {
  return SITE_IDS.filter((id) => sites[id]);
}

export function rowToAnswers(row: ProfileRow): OnboardingAnswers {
  const allowed = new Set(row.allowed_sites);
  return {
    name: row.name,
    roles: row.roles.filter((r): r is Role => ROLE_IDS.has(r)),
    timezone: row.timezone,
    vibe: row.vibe && VIBE_IDS.has(row.vibe) ? (row.vibe as Vibe) : null,
    sites: Object.fromEntries(SITE_IDS.map((id) => [id, allowed.has(id)])) as Record<SiteId, boolean>,
  };
}

// ------------------------------------------------------------
// Kelsie's lines
// ------------------------------------------------------------

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
