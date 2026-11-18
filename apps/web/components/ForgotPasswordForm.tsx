"use client";

import { useActionState } from "react";
import Link from "next/link";
import { requestPasswordReset, type AuthFormState } from "@/app/auth/actions";
import styles from "./AuthPage.module.css";

export default function ForgotPasswordForm({ expired }: { expired: boolean }) {
  const [state, action, pending] = useActionState<AuthFormState, FormData>(
    requestPasswordReset,
    {},
  );

  if (state.sentTo) {
    return (
      <>
        <h1 className={styles.title}>Check your inbox.</h1>
        {/* Worded so it doesn't reveal whether the account exists. */}
        <p className={styles.subtitle}>
          If there&apos;s a Kelsie account for <strong>{state.sentTo}</strong>, we&apos;ve sent
          it a link to reset the password. The link works once and expires soon.
        </p>
        <Link href="/signin" className="btn btn-secondary">
          Back to sign in
        </Link>
      </>
    );
  }

  return (
    <>
      <h1 className={styles.title}>Forgot your password?</h1>
      <p className={styles.subtitle}>
        {expired
          ? "That reset link has expired or was already used. Enter your email for a new one."
          : "No stress. Enter your email and we'll send you a link to set a new one."}
      </p>

      <form className={styles.form} action={action}>
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
        <button type="submit" className={`btn btn-primary ${styles.submit}`} disabled={pending}>
          {pending ? "Sending…" : "Send reset link"}
        </button>
        <p className={styles.error} role="alert">
          {state.error}
        </p>
      </form>

      <p className={styles.switch}>
        Remembered it? <Link href="/signin">Sign in</Link>
      </p>
    </>
  );
}
