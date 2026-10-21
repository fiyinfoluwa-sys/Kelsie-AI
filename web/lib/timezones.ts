/** The browser's timezone, e.g. "America/Toronto". */
export function detectTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

/** "America/Toronto" → "Toronto (America)". */
export function formatTimezone(zone: string): string {
  const parts = zone.split("/");
  if (parts.length < 2) return zone;
  const city = parts[parts.length - 1].replace(/_/g, " ");
  return `${city} (${parts[0]})`;
}

/** Current time in a timezone, e.g. "3:42 PM". */
export function localTime(zone: string, now: Date = new Date()): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      hour: "numeric",
      minute: "2-digit",
      timeZone: zone,
    }).format(now);
  } catch {
    return "";
  }
}

/** All timezones grouped by region, for the dropdown. */
export function timezoneGroups(current: string): [string, string[]][] {
  let zones: string[];
  try {
    zones = Intl.supportedValuesOf("timeZone");
  } catch {
    zones = [];
  }
  // Some browsers report zones (like UTC) that aren't in the list.
  if (current && !zones.includes(current)) zones = [current, ...zones];

  const groups = new Map<string, string[]>();
  for (const zone of zones) {
    const region = zone.includes("/") ? zone.split("/")[0] : "Other";
    groups.set(region, [...(groups.get(region) ?? []), zone]);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([region, list]) => [
      region,
      list.sort((a, b) => formatTimezone(a).localeCompare(formatTimezone(b))),
    ]);
}
