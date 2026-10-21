"use client";

import { useActionState, useState } from "react";
import { deleteAccount, type DeleteState } from "@/app/profile/actions";
import { DELETE_CONFIRMATION } from "@/lib/account";
import styles from "./Profile.module.css";

export default function DeleteAccount() {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [state, action, pending] = useActionState<DeleteState, FormData>(deleteAccount, {});

  if (!open) {
    return (
      <button type="button" className={`btn ${styles.dangerOutline}`} onClick={() => setOpen(true)}>
        Delete my account and data
      </button>
    );
  }

  return (
    <form action={action} className={styles.deleteForm}>
      <p>
        This permanently deletes your account and everything Kelsie has saved for you. It
        can&apos;t be undone.
      </p>
      <label className={styles.deleteField}>
        <span>
          Type <strong>{DELETE_CONFIRMATION}</strong> to confirm
        </span>
        <input
          name="confirm"
          autoComplete="off"
          autoCapitalize="characters"
          spellCheck={false}
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
        />
      </label>
      <div className={styles.deleteActions}>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => {
            setOpen(false);
            setTyped("");
          }}
        >
          Keep my account
        </button>
        <button
          type="submit"
          className={`btn ${styles.danger}`}
          disabled={typed !== DELETE_CONFIRMATION || pending}
        >
          {pending ? "Deleting…" : "Delete forever"}
        </button>
      </div>
      {state.error && (
        <p className={styles.deleteError} role="alert">
          {state.error}
        </p>
      )}
    </form>
  );
}
