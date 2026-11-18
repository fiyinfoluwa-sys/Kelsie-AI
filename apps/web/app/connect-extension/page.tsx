import type { Metadata } from "next";
import AuthCard from "@/components/AuthCard";
import ConnectExtension from "@/components/ConnectExtension";
import { displayName, rowToAnswers } from "@/lib/onboarding";
import { getProfile, requireUser } from "@/lib/session";

export const metadata: Metadata = {
  title: "Connect the extension — Kelsie",
};

export default async function ConnectExtensionPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { supabase, user } = await requireUser();
  const profile = await getProfile(supabase, user.id);
  const name = profile ? displayName(rowToAnswers(profile)) : "friend";

  // The extension opens this page with a random one-time "state" value and
  // only accepts a code that comes back with it.
  const { state } = await searchParams;
  const validState = typeof state === "string" && /^[A-Za-z0-9_-]{22,64}$/.test(state) ? state : null;

  return (
    <AuthCard bubble={`hey ${name}! one click and I'm riding along.`}>
      <ConnectExtension state={validState} email={user.email ?? ""} />
    </AuthCard>
  );
}
