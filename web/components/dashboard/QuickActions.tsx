"use client";

import { useState } from "react";
import styles from "./Dashboard.module.css";

/** Placeholders until the extension and its actions exist. */
export default function QuickActions({ connected }: { connected: boolean }) {
  const [notice, setNotice] = useState<string | null>(null);

  return (
    <div className={styles.quickActions}>
      {connected ? (
        <div className={styles.quickRow}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => setNotice("Page summaries are coming soon.")}
          >
            Summarize a page
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setNotice("Focus mode is coming soon.")}
          >
            Start focus mode
          </button>
        </div>
      ) : (
        <button
          type="button"
          className={`btn btn-primary ${styles.bigButton}`}
          onClick={() => setNotice("The Chrome extension isn't in the Web Store yet. Coming soon!")}
        >
          Add Kelsie to Chrome
        </button>
      )}
      <p className={styles.quickNotice} role="status">
        {notice}
      </p>
    </div>
  );
}
