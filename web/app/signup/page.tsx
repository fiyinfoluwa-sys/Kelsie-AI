import type { Metadata } from "next";
import AuthPage from "@/components/AuthPage";

export const metadata: Metadata = {
  title: "Sign up — Kelsie",
};

export default function SignUpPage() {
  return <AuthPage initialMode="signup" />;
}
