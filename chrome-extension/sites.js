/*
 * The sites a user can allow in Kelsie's Settings, and the web addresses
 * each one covers. The extension asks Chrome for access to exactly these
 * addresses, and only for the sites the user turned on.
 */

export const SITES = [
  {
    id: "google_docs",
    label: "Google Docs and Drive",
    patterns: ["https://docs.google.com/*", "https://drive.google.com/*"],
  },
  {
    id: "canvas",
    label: "Canvas and school sites",
    patterns: ["https://*.instructure.com/*", "https://*.edu/*"],
  },
  {
    id: "gmail",
    label: "Gmail",
    patterns: ["https://mail.google.com/*"],
  },
  {
    id: "linkedin",
    label: "LinkedIn",
    patterns: ["https://www.linkedin.com/*"],
  },
  {
    id: "job_boards",
    label: "Job boards",
    patterns: [
      "https://*.indeed.com/*",
      "https://*.glassdoor.com/*",
      "https://*.greenhouse.io/*",
      "https://*.lever.co/*",
      "https://*.myworkdayjobs.com/*",
      "https://*.wellfound.com/*",
    ],
  },
  {
    id: "everything_else",
    label: "Every other site",
    // All websites. Only requested if the user turned this on.
    patterns: ["*://*/*"],
  },
];

const BY_ID = new Map(SITES.map((site) => [site.id, site]));

export function isSiteId(value) {
  return typeof value === "string" && BY_ID.has(value);
}

export function siteById(id) {
  return BY_ID.get(id);
}

/** Every address pattern for a list of site IDs, without duplicates. */
export function patternsFor(siteIds) {
  return [...new Set(siteIds.flatMap((id) => BY_ID.get(id)?.patterns ?? []))];
}

/**
 * Whether a URL matches a Chrome match pattern like "https://*.edu/*".
 * Only the pattern shapes used above need to work.
 */
export function urlMatches(pattern, href) {
  let url;
  try {
    url = new URL(href);
  } catch {
    return false;
  }
  const match = /^(\*|https?):\/\/([^/]+)(\/.*)$/.exec(pattern);
  if (!match) return false;
  const [, scheme, host] = match;

  const protocol = url.protocol.slice(0, -1);
  if (scheme === "*" ? !["http", "https"].includes(protocol) : protocol !== scheme) return false;

  if (host === "*") return true;
  if (host.startsWith("*.")) {
    const base = host.slice(2);
    return url.hostname === base || url.hostname.endsWith(`.${base}`);
  }
  return url.hostname === host;
}

// ------------------------------------------------------------
// Memory: which pages Kelsie may take notes on
// ------------------------------------------------------------

/*
 * Words that mark a page as sensitive when they appear as a part of the
 * address (a host label or path segment). Kelsie never reads these pages,
 * even on allowed sites.
 */
const SENSITIVE_WORDS = new Set([
  "login", "log-in", "signin", "sign-in", "signup", "sign-up", "auth", "oauth", "sso",
  "password", "passwords", "passwd", "2fa", "mfa", "verify", "security",
  "checkout", "cart", "basket", "payment", "payments", "pay", "billing", "invoice", "invoices",
  "bank", "banking", "onlinebanking", "wallet", "paypal", "stripe", "venmo", "crypto",
  "health", "medical", "patient", "patients", "clinic", "pharmacy", "therapy", "mychart",
  "insurance", "tax", "taxes",
]);
const SENSITIVE_HOSTS = ["accounts.google.com", "myaccount.google.com", "pay.google.com", "payments.google.com"];

export function isSensitiveUrl(href) {
  let url;
  try {
    url = new URL(href);
  } catch {
    return true;
  }
  if (SENSITIVE_HOSTS.includes(url.hostname)) return true;
  const parts = [...url.hostname.split("."), ...url.pathname.toLowerCase().split(/[/._]+/)];
  return parts.some((part) => SENSITIVE_WORDS.has(part) || /bank|checkout|password/.test(part));
}

const TRACKING_PARAMS = /^(utm_.*|fbclid|gclid|dclid|msclkid|mc_cid|mc_eid|ref|ref_src|si|igshid)$/i;

/**
 * The address used to recognise "the same page" again: no tracking
 * parameters, and no #fragment unless it's an app route like Gmail's
 * "#inbox/<message>". Only a hash of this ever leaves the extension.
 */
export function normalizeUrl(href) {
  const url = new URL(href);
  const params = [...url.searchParams.entries()]
    .filter(([key]) => !TRACKING_PARAMS.test(key))
    .sort(([a], [b]) => a.localeCompare(b));
  const query = params.length ? `?${new URLSearchParams(params)}` : "";
  const hash = url.hash.includes("/") ? url.hash : "";
  return `${url.protocol}//${url.hostname.toLowerCase()}${url.pathname}${query}${hash}`;
}

/** Which allowed site a page belongs to (a specific site beats "Every other site"). */
export function siteForUrl(allowedSiteIds, href) {
  const matching = SITES.filter(
    (site) => allowedSiteIds.includes(site.id) && site.patterns.some((p) => urlMatches(p, href)),
  );
  return (matching.find((s) => s.id !== "everything_else") ?? matching[0])?.id ?? null;
}
