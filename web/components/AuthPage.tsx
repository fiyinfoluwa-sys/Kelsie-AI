"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Orb from "./Orb";
import SpeechBubble from "./SpeechBubble";
import SplitScreen from "./SplitScreen";
import { GoogleIcon } from "./icons";
import styles from "./AuthPage.module.css";

type Mode = "signup" | "signin";

const COPY: Record<
  Mode,
  { title: string; subtitle: string; submit: string; path: string }
> = {
  signup: {
    title: "Let's get you set up.",
    subtitle: "Make an account and Kelsie starts riding along in under a minute.",
    submit: "Create account",
    path: "/signup",
  },
  signin: {
    title: "Welcome back.",
    subtitle: "Kelsie kept your spot. Sign in to pick up where you left off.",
    submit: "Sign in",
    path: "/signin",
  },
};

const STEPS = [
  {
    title: "Make your account",
    body: "Google or email, whatever's easiest.",
  },
  {
    title: "Tell Kelsie about you",
    body: "Your name, what you're up to, your timezone and your vibe.",
  },
  {
    title: "Add the Chrome extension",
    body: "It links to your account so Kelsie remembers across tabs.",
  },
];

export default function AuthPage({ initialMode }: { initialMode: Mode }) {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>(initialMode);
  const [showPassword, setShowPassword] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const copy = COPY[mode];

  function switchMode(next: Mode) {
    setMode(next);
    setNotice(null);
    // Keep the URL shareable without a full page navigation.
    window.history.replaceState(null, "", COPY[next].path);
  }

  // Real accounts aren't wired up yet: sign-up goes straight to
  // onboarding, and sign-in explains that it's a preview.
  function continueWithGoogle() {
    if (mode === "signup") {
      router.push("/onboarding");
    } else {
      showPreviewNotice();
    }
  }

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const email = form.elements.namedItem("email") as HTMLInputElement;
    const password = form.elements.namedItem("password") as HTMLInputElement;

    if (!email.validity.valid) {
      setNotice("That email doesn't look quite right.");
      email.focus();
      return;
    }
    if (!password.validity.valid) {
      setNotice("Passwords need at least 8 characters.");
      password.focus();
      return;
    }

    if (mode === "signup") {
      router.push("/onboarding");
    } else {
      showPreviewNotice();
    }
  }

  function showPreviewNotice() {
    setNotice("Sign-in isn't hooked up yet. This page is a preview for now.");
  }

  const panel = (
    <>
      <div className={styles.panelArt}>
        <Orb size={150} />
        <SpeechBubble tone="dark" className={styles.panelBubble}>
          three quick steps and I&apos;m all yours.
        </SpeechBubble>
      </div>

      <ol className={styles.steps}>
        {STEPS.map((step, index) => (
          <li key={step.title} className={styles.step}>
            <span className={styles.stepNumber}>{index + 1}</span>
            <div>
              <h2 className={styles.stepTitle}>{step.title}</h2>
              <p className={styles.stepBody}>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>
    </>
  );

  return (
    <SplitScreen panel={panel} panelLabel="How setup works">
      <div className={styles.toggle} role="group" aria-label="Account">
        <button
          type="button"
          aria-pressed={mode === "signup"}
          onClick={() => switchMode("signup")}
        >
          Sign up
        </button>
        <button
          type="button"
          aria-pressed={mode === "signin"}
          onClick={() => switchMode("signin")}
        >
          Sign in
        </button>
      </div>

      <h1 className={styles.title}>{copy.title}</h1>
      <p className={styles.subtitle}>{copy.subtitle}</p>

      <button
        type="button"
        className={`btn ${styles.google}`}
        onClick={continueWithGoogle}
      >
        <GoogleIcon />
        Continue with Google
      </button>

      <div className={styles.divider}>
        <span>or use email</span>
      </div>

      <form className={styles.form} onSubmit={submit} noValidate>
        <label className={styles.field}>
          <span>Email</span>
          <input
            type="email"
            name="email"
            autoComplete="email"
            placeholder="you@school.edu"
            required
          />
        </label>

        <label className={styles.field}>
          <span className={styles.labelRow}>
            Password
            {mode === "signin" && (
              <button
                type="button"
                className={styles.forgot}
                onClick={showPreviewNotice}
              >
                Forgot it?
              </button>
            )}
          </span>
          <span className={styles.passwordWrap}>
            <input
              type={showPassword ? "text" : "password"}
              name="password"
              autoComplete={
                mode === "signup" ? "new-password" : "current-password"
              }
              placeholder={
                mode === "signup" ? "At least 8 characters" : "Your password"
              }
              minLength={8}
              required
            />
            <button
              type="button"
              className={styles.reveal}
              onClick={() => setShowPassword((shown) => !shown)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              {showPassword ? "Hide" : "Show"}
            </button>
          </span>
        </label>

        <button type="submit" className={`btn btn-primary ${styles.submit}`}>
          {copy.submit}
        </button>

        <p className={styles.notice} role="status">
          {notice}
        </p>
      </form>

      <p className={styles.switch}>
        {mode === "signup" ? "Already have an account? " : "New here? "}
        <button
          type="button"
          onClick={() => switchMode(mode === "signup" ? "signin" : "signup")}
        >
          {mode === "signup" ? "Sign in" : "Create one"}
        </button>
      </p>
    </SplitScreen>
  );
}
