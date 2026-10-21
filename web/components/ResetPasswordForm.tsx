"use client";

import { useActionState, useState } from "react";
import { updatePassword, type AuthFormState } from "@/app/auth/actions";
import { PASSWORD_MAX, PASSWORD_MIN } from "@/lib/auth-messages";
import styles from "./AuthPage.module.css";

export default function ResetPasswordForm() {
  const [state, action, pending] = useActionState<AuthFormState, FormData>(updatePassword, {});
  const [show, setShow] = useState(false);

  return (
    <>
      <h1 className={styles.title}>Set a new password.</h1>
      <p className={styles.subtitle}>Make it at least {PASSWORD_MIN} characters.</p>

      <form className={styles.form} action={action}>
        <label className={styles.field}>
          <span>New password</span>
          <span className={styles.passwordWrap}>
            <input
              type={show ? "text" : "password"}
              name="password"
              autoComplete="new-password"
              minLength={PASSWORD_MIN}
              maxLength={PASSWORD_MAX}
              required
            />
            <button
              type="button"
              className={styles.reveal}
              onClick={() => setShow((s) => !s)}
              aria-label={show ? "Hide passwords" : "Show passwords"}
            >
              {show ? "Hide" : "Show"}
            </button>
          </span>
        </label>
        <label className={styles.field}>
          <span>Type it again</span>
          <input
            type={show ? "text" : "password"}
            name="confirm"
            autoComplete="new-password"
            minLength={PASSWORD_MIN}
            maxLength={PASSWORD_MAX}
            required
          />
        </label>
        <button type="submit" className={`btn btn-primary ${styles.submit}`} disabled={pending}>
          {pending ? "Saving…" : "Save new password"}
        </button>
        <p className={styles.error} role="alert">
          {state.error}
        </p>
      </form>
    </>
  );
}
