import { NextResponse } from "next/server";
import { createClient } from "@/lib/supabase/server";

/**
 * "Download my data": everything Kelsie stores for the signed-in user, as
 * JSON. Row Level Security limits every query to their own rows.
 *
 * Another site can make a browser request this URL, but it can't read the
 * response (browsers block cross-site reads), and SameSite=Lax cookies
 * aren't sent on cross-site background requests anyway.
 */
export async function GET() {
  const supabase = await createClient();
  const { data: auth } = await supabase.auth.getUser();
  const user = auth?.user;
  if (!user) return NextResponse.json({ error: "Sign in first." }, { status: 401 });

  const [profile, memories, deadlines, applications] = await Promise.all([
    supabase
      .from("profiles")
      .select(
        "name, roles, timezone, vibe, allowed_sites, memory_enabled, onboarding_completed_at, extension_connected_at, created_at, updated_at",
      )
      .eq("id", user.id)
      .maybeSingle(),
    supabase
      .from("memories")
      .select("content, source, created_at, updated_at")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false }),
    supabase
      .from("deadlines")
      .select("title, details, due_date, status, source, created_at")
      .eq("user_id", user.id)
      .order("due_date"),
    supabase
      .from("applications")
      .select("company, role, status, applied_on, created_at")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false }),
  ]);

  if (profile.error || memories.error || deadlines.error || applications.error) {
    return NextResponse.json({ error: "Couldn't export your data. Try again." }, { status: 500 });
  }

  const body = {
    exported_at: new Date().toISOString(),
    account: {
      email: user.email,
      signed_in_with: user.app_metadata?.provider ?? "email",
      created_at: user.created_at,
    },
    profile: profile.data,
    memories: memories.data,
    deadlines: deadlines.data,
    applications: applications.data,
  };

  const date = new Date().toISOString().slice(0, 10);
  return new NextResponse(JSON.stringify(body, null, 2), {
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Content-Disposition": `attachment; filename="kelsie-data-${date}.json"`,
      // Personal data: never cache it anywhere.
      "Cache-Control": "no-store, private",
    },
  });
}
