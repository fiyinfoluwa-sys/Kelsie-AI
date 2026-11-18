import type { Metadata } from "next";
import AuthCard from "@/components/AuthCard";
import ForgotPasswordForm from "@/components/ForgotPasswordForm";

export const metadata: Metadata = {
  title: "Reset your password — Kelsie",
};

export default async function ForgotPasswordPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { expired } = await searchParams;
  return (
    <AuthCard bubble="happens to everyone. let's get you back in.">
      <ForgotPasswordForm expired={Boolean(expired)} />
    </AuthCard>
  );
}
