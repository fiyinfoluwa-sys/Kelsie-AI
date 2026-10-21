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
