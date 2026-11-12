import { revalidatePath } from "next/cache";
import { setSiteAllowed } from "@/app/profile/actions";
import { clearMemories } from "@/app/settings/actions";
import { daysBetween, localDate } from "@/lib/dates";
import { loadContext } from "@/lib/kelsie/context";
import { isSameOrigin, json } from "@/lib/kelsie/request";
import { checkSiteId } from "@/lib/onboarding";

/**
 * POST { action, ... } — carries out a confirm step from /api/kelsie/message
 * after the user said yes in the UI. Everything is validated again here:
 * nothing from the earlier reply is trusted.
 *
 * Deleting the account is deliberately not possible here. It stays behind
 * Settings' "type DELETE" step.
 */
export async function POST(request: Request) {
  if (!isSameOrigin(request)) return json({ error: "forbidden" }, 403);
  const ctx = await loadContext();
  if (!ctx) return json({ error: "not_signed_in" }, 401);

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return json({ error: "bad_request" }, 400);
  }
  if (typeof body !== "object" || body === null) return json({ error: "bad_request" }, 400);

  switch (body.action) {
    case "delete_account":
      return json({ error: "use_settings", href: "/settings#delete-title" }, 400);

    case "clear_memories": {
      const result = await clearMemories();
      return result.ok ? json({ ok: true }) : json({ error: result.error }, 500);
    }

    case "remove_site":
    case "add_site": {
      const site = checkSiteId(body.site);
      if (!site.ok) return json({ error: "bad_request" }, 400);
      const result = await setSiteAllowed(site.value, body.action === "add_site");
      return result.ok ? json({ ok: true, sites: result.sites }) : json({ error: result.error }, 500);
    }

    case "add_deadline": {
      const title =
        typeof body.title === "string" ? body.title.replace(/[\u0000-\u001f<>]/g, " ").replace(/\s+/g, " ").trim() : "";
      const dueDate = typeof body.dueDate === "string" ? body.dueDate : "";
      const today = localDate(ctx.timezone);
      const validDate =
        /^\d{4}-\d{2}-\d{2}$/.test(dueDate) &&
        !Number.isNaN(Date.parse(`${dueDate}T00:00:00Z`)) &&
        daysBetween(today, dueDate) >= -1 &&
        daysBetween(today, dueDate) <= 400;
      if (!title || title.length > 120 || !validDate) return json({ error: "bad_request" }, 400);

      const { error } = await ctx.supabase.from("deadlines").insert({
        user_id: ctx.userId,
        title,
        details: "",
        due_date: dueDate,
        status: "confirmed",
        source: "other",
      });
      if (error) return json({ error: "Couldn't save that deadline." }, 500);
      revalidatePath("/profile");
      return json({ ok: true });
    }

    default:
      return json({ error: "bad_request" }, 400);
  }
}
