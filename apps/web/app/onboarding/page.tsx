import type { Metadata } from "next";
import Onboarding from "@/components/Onboarding";
import { NAME_MAX_LENGTH, defaultAnswers, rowToAnswers } from "@/lib/onboarding";
import { getProfile, requireUser } from "@/lib/session";

export const metadata: Metadata = {
  title: "Set up Kelsie",
};

export default async function OnboardingPage() {
  // Redirects to sign-in unless there's a verified, email-confirmed user.
  const { supabase, user } = await requireUser();
  const profile = await getProfile(supabase, user.id);

  let initialAnswers = profile ? rowToAnswers(profile) : defaultAnswers();
  if (!profile) {
    // Google accounts come with a name; use the first one as a starting point.
    const fullName = user.user_metadata?.full_name ?? user.user_metadata?.name;
    if (typeof fullName === "string") {
      initialAnswers = {
        ...initialAnswers,
        name: fullName.trim().split(/\s+/)[0]?.slice(0, NAME_MAX_LENGTH) ?? "",
      };
    }
  }

  return (
    <Onboarding
      initialAnswers={initialAnswers}
      completed={Boolean(profile?.onboarding_completed_at)}
    />
  );
}
