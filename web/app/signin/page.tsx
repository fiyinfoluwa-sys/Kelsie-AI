import type { Metadata } from "next";
import AuthPage from "@/components/AuthPage";

export const metadata: Metadata = {
  title: "Sign in — Kelsie",
};

export default function SignInPage() {
  return <AuthPage initialMode="signin" />;
}
