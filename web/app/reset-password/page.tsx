import type { Metadata } from "next";
import { redirect } from "next/navigation";
import AuthCard from "@/components/AuthCard";
import ResetPasswordForm from "@/components/ResetPasswordForm";
import { cameFromResetLink } from "@/lib/session";
import { createClient } from "@/lib/supabase/server";

export const metadata: Metadata = {
  title: "Set a new password — Kelsie",
};

export default async function ResetPasswordPage() {
  // Only reachable straight after opening a reset link: an ordinary
  // signed-in session can't change the password without the old one.
  const supabase = await createClient();
  const { data } = await supabase.auth.getClaims();
  if (!data?.claims) redirect("/signin");
  if (!cameFromResetLink(data.claims.amr)) redirect("/forgot-password?expired=1");

  return (
    <AuthCard bubble="fresh start. pick something you'll remember.">
      <ResetPasswordForm />
    </AuthCard>
  );
}
