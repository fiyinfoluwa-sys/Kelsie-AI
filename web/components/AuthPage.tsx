"use client";

import { useState } from "react";
import Logo from "./Logo";
import Orb from "./Orb";
import SpeechBubble from "./SpeechBubble";
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

  function preview(event?: React.FormEvent) {
    event?.preventDefault();
    setNotice(
      "Sign-in isn't hooked up yet. This page is a preview for now."
    );
  }

  return (
    <div className={styles.page}>
      <main className={styles.formSide}>
        <div className={styles.formTop}>
          <Logo />
        </div>

        <div className={styles.formWrap}>
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
            onClick={() => preview()}
          >
            <GoogleIcon />
            Continue with Google
          </button>

          <div className={styles.divider}>
            <span>or use email</span>
          </div>

          <form className={styles.form} onSubmit={preview} noValidate>
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
                    onClick={() => preview()}
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
              onClick={() =>
                switchMode(mode === "signup" ? "signin" : "signup")
              }
            >
              {mode === "signup" ? "Sign in" : "Create one"}
            </button>
          </p>
        </div>
      </main>

      <aside className={styles.panel} aria-label="How setup works">
        <div className={styles.panelInner}>
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
        </div>
      </aside>
    </div>
  );
}
