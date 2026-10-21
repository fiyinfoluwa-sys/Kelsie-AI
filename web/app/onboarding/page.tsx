import type { Metadata } from "next";
import Onboarding from "@/components/Onboarding";

export const metadata: Metadata = {
  title: "Set up Kelsie",
};

export default function OnboardingPage() {
  return <Onboarding />;
}
