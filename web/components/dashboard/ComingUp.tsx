"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { reviewDeadline } from "@/app/profile/actions";
import styles from "./Dashboard.module.css";

export type UpcomingItem = {
  id: string;
  title: string;
  details: string;
  month: string;
  day: string;
  tag: "Today" | "Tomorrow" | null;
};
export type SuggestedItem = { id: string; title: string; details: string; when: string };

export default function ComingUp({
  upcoming,
  suggested,
}: {
  upcoming: UpcomingItem[];
  suggested: SuggestedItem[];
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  function review(id: string, decision: "confirmed" | "dismissed") {
    startTransition(async () => {
      const result = await reviewDeadline(id, decision);
      if (!result.ok) {
        setError(result.error);
        return;
      }
      setError(null);
      // Re-read from the server so a confirmed deadline moves into the list.
      router.refresh();
    });
  }

  const empty = upcoming.length === 0 && suggested.length === 0;

  return (
    <section className={`${styles.card} ${styles.comingCard}`} aria-labelledby="coming-title">
      <div className={styles.cardHead}>
        <h2 id="coming-title">Coming up</h2>
      </div>

      {empty ? (
        <div className={styles.empty}>
          <h3>No deadlines yet</h3>
          <p>
            When Kelsie spots a due date on Canvas, in a syllabus or in your inbox, it&apos;ll ask
            you to confirm it, then it shows up here.
          </p>
        </div>
      ) : (
        <>
          {upcoming.length > 0 && (
            <ul className={styles.deadlines}>
              {upcoming.map((d) => (
                <li key={d.id} className={styles.deadline}>
                  <span className={styles.dateTile} aria-hidden="true">
                    <span className={styles.tileMonth}>{d.month}</span>
                    <span className={styles.tileDay}>{d.day}</span>
                  </span>
                  <div className={styles.deadlineText}>
                    <p className={styles.deadlineTitle}>
                      {d.title}
                      {d.tag && <span className={styles.tag}>{d.tag}</span>}
                    </p>
                    <p className={styles.meta}>
                      <span className="visually-hidden">
                        Due {d.month} {d.day}.{" "}
                      </span>
                      {d.details}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          )}

          {suggested.map((s) => (
            <div key={s.id} className={styles.suggestion}>
              <p className={styles.suggestionLabel}>Kelsie spotted this. Is it a deadline?</p>
              <p className={styles.deadlineTitle}>{s.title}</p>
              <p className={styles.meta}>
                {s.when}
                {s.details ? ` · ${s.details}` : ""}
              </p>
              <div className={styles.suggestionActions}>
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  onClick={() => review(s.id, "confirmed")}
                  disabled={pending}
                >
                  Add it
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => review(s.id, "dismissed")}
                  disabled={pending}
                >
                  Not a deadline
                </button>
              </div>
            </div>
          ))}
        </>
      )}

      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
