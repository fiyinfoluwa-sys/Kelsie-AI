import type { Metadata } from "next";
import AuthPage from "@/components/AuthPage";
import { redirectMessage } from "@/lib/auth-redirect-messages";

export const metadata: Metadata = {
  title: "Sign up — Kelsie",
};

export default async function SignUpPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  return <AuthPage initialMode="signup" initialMessage={redirectMessage(await searchParams)} />;
}
