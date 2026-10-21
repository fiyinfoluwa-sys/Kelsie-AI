"use client";

import { useActionState, useState } from "react";
import Link from "next/link";
import Orb from "./Orb";
import SpeechBubble from "./SpeechBubble";
import SplitScreen from "./SplitScreen";
import { GoogleIcon } from "./icons";
import { signIn, signInWithGoogle, signUp, type AuthFormState } from "@/app/auth/actions";
import { PASSWORD_MAX, PASSWORD_MIN } from "@/lib/auth-messages";
import styles from "./AuthPage.module.css";

type Mode = "signup" | "signin";

const COPY: Record<
  Mode,
  { title: string; subtitle: string; submit: string; pending: string; path: string }
> = {
  signup: {
    title: "Let's get you set up.",
    subtitle: "Make an account and Kelsie starts riding along in under a minute.",
    submit: "Create account",
    pending: "Creating account…",
    path: "/signup",
  },
  signin: {
    title: "Welcome back.",
    subtitle: "Kelsie kept your spot. Sign in to pick up where you left off.",
    submit: "Sign in",
    pending: "Signing in…",
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

type AuthPageProps = {
  initialMode: Mode;
  /** A message from a redirect, e.g. an expired email link. */
  initialMessage?: string;
};

export default function AuthPage({ initialMode, initialMessage }: AuthPageProps) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [showPassword, setShowPassword] = useState(false);
  const [signUpState, signUpAction, signingUp] = useActionState<AuthFormState, FormData>(
    signUp,
    {},
  );
  const [signInState, signInAction, signingIn] = useActionState<AuthFormState, FormData>(
    signIn,
    {},
  );
  const [message, setMessage] = useState(initialMessage);

  const copy = COPY[mode];
  const state = mode === "signup" ? signUpState : signInState;
  const pending = mode === "signup" ? signingUp : signingIn;

  function switchMode(next: Mode) {
    setMode(next);
    setMessage(undefined);
    // Keep the URL shareable without a full page navigation.
    window.history.replaceState(null, "", COPY[next].path);
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

  // After sign-up: same screen whether or not the email already had an
  // account, so it can't be used to check who's registered.
  if (mode === "signup" && signUpState.sentTo) {
    return (
      <SplitScreen panel={panel} panelLabel="How setup works">
        <h1 className={styles.title}>Check your inbox.</h1>
        <p className={styles.subtitle}>
          We sent a link to <strong>{signUpState.sentTo}</strong>. Open it to verify your
          email and finish setting up. It can take a minute to arrive.
        </p>
        <p className={styles.subtitle}>
          Already have an account?{" "}
          <button type="button" className={styles.inlineLink} onClick={() => switchMode("signin")}>
            Sign in instead
          </button>
        </p>
      </SplitScreen>
    );
  }

  const error = state.error ?? message;

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

      <form action={signInWithGoogle}>
        <button type="submit" className={`btn ${styles.google}`}>
          <GoogleIcon />
          Continue with Google
        </button>
      </form>

      <div className={styles.divider}>
        <span>or use email</span>
      </div>

      <form
        key={mode}
        className={styles.form}
        action={mode === "signup" ? signUpAction : signInAction}
      >
        <label className={styles.field}>
          <span>Email</span>
          <input
            type="email"
            name="email"
            autoComplete="email"
            placeholder="you@school.edu"
            maxLength={254}
            required
          />
        </label>

        <label className={styles.field}>
          <span className={styles.labelRow}>
            Password
            {mode === "signin" && (
              <Link href="/forgot-password" className={styles.forgot}>
                Forgot it?
              </Link>
            )}
          </span>
          <span className={styles.passwordWrap}>
            <input
              type={showPassword ? "text" : "password"}
              name="password"
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
              placeholder={
                mode === "signup" ? `At least ${PASSWORD_MIN} characters` : "Your password"
              }
              minLength={mode === "signup" ? PASSWORD_MIN : undefined}
              maxLength={PASSWORD_MAX}
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

        <button
          type="submit"
          className={`btn btn-primary ${styles.submit}`}
          disabled={pending}
        >
          {pending ? copy.pending : copy.submit}
        </button>

        <p className={styles.error} role="alert">
          {error}
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
