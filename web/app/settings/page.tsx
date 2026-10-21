import type { Metadata } from "next";
import { redirect } from "next/navigation";
import AppNav from "@/components/AppNav";
import SettingsForm from "@/components/settings/SettingsForm";
import { showsApplications } from "@/lib/content";
import { displayName, rowToAnswers } from "@/lib/onboarding";
import { getProfile, requireUser } from "@/lib/session";

export const metadata: Metadata = {
  title: "Settings — Kelsie",
};

export default async function SettingsPage() {
  const { supabase, user } = await requireUser();
  const profile = await getProfile(supabase, user.id);
  if (!profile?.onboarding_completed_at) redirect("/onboarding");

  // Counts only (no rows): for the nav and the "Clear all memories" prompt.
  const [applications, memories] = await Promise.all([
    supabase.from("applications").select("id", { count: "exact", head: true }).eq("user_id", user.id),
    supabase.from("memories").select("id", { count: "exact", head: true }).eq("user_id", user.id),
  ]);

  const answers = rowToAnswers(profile);
  const provider = user.app_metadata?.provider === "google" ? "google" : "email";

  return (
    <>
      <AppNav
        active="settings"
        name={displayName(answers)}
        showApplications={showsApplications(answers.roles, applications.count ?? 0)}
      />
      <SettingsForm
        initial={answers}
        memoryEnabled={profile.memory_enabled}
        memoryCount={memories.count ?? 0}
        email={user.email ?? ""}
        provider={provider}
      />
    </>
  );
}
