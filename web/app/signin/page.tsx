import type { Metadata } from "next";
import AuthPage from "@/components/AuthPage";
import { redirectMessage } from "@/lib/auth-redirect-messages";

export const metadata: Metadata = {
  title: "Sign in — Kelsie",
};

export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  return <AuthPage initialMode="signin" initialMessage={redirectMessage(await searchParams)} />;
}
