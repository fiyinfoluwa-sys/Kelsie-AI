"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { createExtensionCode } from "@/app/connect-extension/actions";
import { EXTENSION_ID, type ChromeRuntime } from "@/lib/extension";
import styles from "./AuthPage.module.css";

type Outcome = "idle" | "connected" | "error";

export default function ConnectExtension({ state, email }: { state: string | null; email: string }) {
  const [outcome, setOutcome] = useState<Outcome>("idle");
  const [message, setMessage] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  if (!state) {
    return (
      <>
        <h1 className={styles.title}>Start from the extension.</h1>
        <p className={styles.subtitle}>
          Click the Kelsie icon in your Chrome toolbar, then <strong>Connect to Kelsie</strong>.
          This page only works when the extension opens it.
        </p>
      </>
    );
  }

  function connect() {
    const runtime = (window as unknown as { chrome?: { runtime?: ChromeRuntime } }).chrome?.runtime;
    if (!runtime?.sendMessage) {
      setOutcome("error");
      setMessage("Kelsie's extension isn't installed in this browser, or it's turned off.");
      return;
    }

    startTransition(async () => {
      // Only made after this click, so nothing happens without your say-so.
      const result = await createExtensionCode();
      if (!result.ok) {
        setOutcome("error");
        setMessage(result.error);
        return;
      }
      await new Promise<void>((resolve) => {
        // Sent only to Kelsie's extension ID. Chrome delivers it only if
        // that extension lists this site in externally_connectable.
        runtime.sendMessage(EXTENSION_ID, { type: "kelsie-connect", code: result.code, state }, (reply) => {
          const ok = (reply as { ok?: boolean } | undefined)?.ok === true;
          if (runtime.lastError || !ok) {
            setOutcome("error");
            setMessage(
              (reply as { error?: string } | undefined)?.error ??
                "The extension didn't answer. Make sure it's installed, then try again from its icon.",
            );
          } else {
            setOutcome("connected");
            setMessage(null);
          }
          resolve();
        });
      });
    });
  }

  if (outcome === "connected") {
    return (
      <>
        <h1 className={styles.title}>You&apos;re connected.</h1>
        <p className={styles.subtitle}>
          Kelsie&apos;s extension is now linked to <strong>{email}</strong>. A tab will open so you can
          let Kelsie onto the sites you picked. You can close this one.
        </p>
        <Link href="/profile" className="btn btn-secondary">
          Back to my profile
        </Link>
      </>
    );
  }

  return (
    <>
      <h1 className={styles.title}>Connect Kelsie to Chrome.</h1>
      <p className={styles.subtitle}>
        This links the extension in this browser to <strong>{email}</strong>. It can see your name,
        vibe and the sites you&apos;ve allowed, and nothing else. You can disconnect anytime.
      </p>
      <button
        type="button"
        className={`btn btn-primary ${styles.submit}`}
        onClick={connect}
        disabled={pending}
      >
        {pending ? "Connecting…" : "Connect this browser"}
      </button>
      <p className={styles.error} role="alert">
        {outcome === "error" ? message : null}
      </p>
    </>
  );
}
