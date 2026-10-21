/*
 * Dates shown in the user's own timezone (not the server's).
 */

/** "2026-10-21" for the given moment in a timezone. */
export function localDate(zone: string, at: Date = new Date()): string {
  // en-CA formats as YYYY-MM-DD.
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: zone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(at);
}

export function localHour(zone: string, at: Date = new Date()): number {
  const hour = new Intl.DateTimeFormat("en-US", {
    timeZone: zone,
    hour: "numeric",
    hourCycle: "h23",
  }).format(at);
  return Number(hour);
}

/** "Wednesday, October 21" in the user's timezone. */
export function longDate(zone: string, at: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: zone,
    weekday: "long",
    month: "long",
    day: "numeric",
  }).format(at);
}

/** Whole days from one YYYY-MM-DD date to another. */
export function daysBetween(from: string, to: string): number {
  const ms = Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`);
  return Math.round(ms / 86_400_000);
}

/** "Today", "Yesterday", "3 days ago" or "Oct 12" for a past timestamp. */
export function relativeDay(timestamp: string, zone: string, now: Date = new Date()): string {
  const days = daysBetween(localDate(zone, new Date(timestamp)), localDate(zone, now));
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  return shortDate(localDate(zone, new Date(timestamp)));
}

/** "Oct 12" for a YYYY-MM-DD date (no timezone shift). */
export function shortDate(date: string): string {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" }).format(
    new Date(`${date}T00:00:00Z`),
  );
}

/** { month: "OCT", day: "22" } for a date tile. */
export function dateTile(date: string): { month: string; day: string } {
  const d = new Date(`${date}T00:00:00Z`);
  return {
    month: new Intl.DateTimeFormat("en-US", { month: "short", timeZone: "UTC" }).format(d).toUpperCase(),
    day: String(d.getUTCDate()),
  };
}
