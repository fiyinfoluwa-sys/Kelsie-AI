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

export type Checked<T> = { ok: true; value: T } | { ok: false; error: string };
export type ValidationResult = Checked<OnboardingAnswers>;

const SITES_ERROR = "Something went wrong with your sites. Please try again.";

/*
 * One validator per field, so Settings can check a single change with the
 * same rules onboarding uses. Every value is untrusted: it comes from the
 * browser.
 */

export function checkName(input: unknown): Checked<string> {
  if (typeof input !== "string") return { ok: false, error: "Please enter a name." };
  // Collapse whitespace and drop control characters.
  const name = input.replace(/[\u0000-\u001f\u007f]/g, "").replace(/\s+/g, " ").trim();
  if (name.length > NAME_MAX_LENGTH) {
    return { ok: false, error: `Names can be up to ${NAME_MAX_LENGTH} characters.` };
  }
  return { ok: true, value: name };
}

export function checkRoles(input: unknown): Checked<Role[]> {
  if (
    !Array.isArray(input) ||
    input.length === 0 ||
    input.length > 3 ||
    !input.every((r) => typeof r === "string" && ROLE_IDS.has(r)) ||
    new Set(input).size !== input.length ||
    (input.includes("mix") && input.length > 1)
  ) {
    return { ok: false, error: "Pick what you're up to." };
  }
  return { ok: true, value: input as Role[] };
}

export function checkTimezone(input: unknown): Checked<string> {
  const valid = (() => {
    if (typeof input !== "string" || input.length < 1 || input.length > 64) return false;
    if (!/^[A-Za-z0-9_+-]+(\/[A-Za-z0-9_+-]+)*$/.test(input)) return false;
    try {
      // Throws a RangeError for anything that isn't a real IANA zone.
      new Intl.DateTimeFormat("en-US", { timeZone: input });
      return true;
    } catch {
      return false;
    }
  })();
  return valid ? { ok: true, value: input as string } : { ok: false, error: "Pick a valid timezone." };
}

export function checkVibe(input: unknown): Checked<Vibe> {
  return typeof input === "string" && VIBE_IDS.has(input)
    ? { ok: true, value: input as Vibe }
    : { ok: false, error: "Pick a vibe for Kelsie." };
}

/** A full on/off map with every site present. */
export function checkSites(input: unknown): Checked<Record<SiteId, boolean>> {
  if (typeof input !== "object" || input === null) return { ok: false, error: SITES_ERROR };
  const raw = input as Record<string, unknown>;
  const sites = { ...DEFAULT_SITES };
  for (const id of SITE_IDS) {
    if (typeof raw[id] !== "boolean") return { ok: false, error: SITES_ERROR };
    sites[id] = raw[id] as boolean;
  }
  return { ok: true, value: sites };
}

export function checkSiteId(input: unknown): Checked<SiteId> {
  return typeof input === "string" && (SITE_IDS as string[]).includes(input)
    ? { ok: true, value: input as SiteId }
    : { ok: false, error: SITES_ERROR };
}

/**
 * Checks a whole set of onboarding answers and returns a clean copy
 * containing only known fields with allowed values.
 */
export function validateAnswers(input: unknown): ValidationResult {
  if (typeof input !== "object" || input === null) {
    return { ok: false, error: "Something went wrong. Please try again." };
  }
  const raw = input as Record<string, unknown>;
  const name = checkName(raw.name);
  if (!name.ok) return name;
  const roles = checkRoles(raw.roles);
  if (!roles.ok) return roles;
  const timezone = checkTimezone(raw.timezone);
  if (!timezone.ok) return timezone;
  const vibe = checkVibe(raw.vibe);
  if (!vibe.ok) return vibe;
  const sites = checkSites(raw.sites);
  if (!sites.ok) return sites;
  return {
    ok: true,
    value: {
      name: name.value,
      roles: roles.value,
      timezone: timezone.value,
      vibe: vibe.value,
      sites: sites.value,
    },
  };
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
  memory_enabled: boolean;
  extension_connected_at: string | null;
};

export function sitesToList(sites: Record<SiteId, boolean>): SiteId[] {
  return SITE_IDS.filter((id) => sites[id]);
}

export function sitesFromList(list: string[]): Record<SiteId, boolean> {
  const allowed = new Set(list);
  return Object.fromEntries(SITE_IDS.map((id) => [id, allowed.has(id)])) as Record<SiteId, boolean>;
}

export function rowToAnswers(row: ProfileRow): OnboardingAnswers {
  return {
    name: row.name,
    roles: row.roles.filter((r): r is Role => ROLE_IDS.has(r)),
    timezone: row.timezone,
    vibe: row.vibe && VIBE_IDS.has(row.vibe) ? (row.vibe as Vibe) : null,
    sites: sitesFromList(row.allowed_sites),
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
