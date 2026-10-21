"use client";

import { useState, useTransition } from "react";
import { deleteMemory, updateMemory } from "@/app/profile/actions";
import { updateSettings } from "@/app/settings/actions";
import { PencilIcon, TrashIcon } from "@/components/icons";
import { MEMORY_MAX_LENGTH } from "@/lib/content";
import styles from "./Dashboard.module.css";

export type TimelineItem = { id: string; content: string; meta: string };

export default function MemoryTimeline({
  items: initialItems,
  memoryEnabled: initiallyEnabled,
}: {
  items: TimelineItem[];
  memoryEnabled: boolean;
}) {
  const [items, setItems] = useState(initialItems);
  const [enabled, setEnabled] = useState(initiallyEnabled);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  function startEditing(item: TimelineItem) {
    setEditingId(item.id);
    setDraft(item.content);
    setError(null);
  }

  function save(id: string) {
    startTransition(async () => {
      const result = await updateMemory(id, draft);
      if (!result.ok) {
        setError(result.error);
        return;
      }
      setItems((current) =>
        current.map((item) => (item.id === id ? { ...item, content: draft.trim() } : item)),
      );
      setEditingId(null);
      setError(null);
    });
  }

  function remove(id: string) {
    startTransition(async () => {
      const result = await deleteMemory(id);
      if (!result.ok) {
        setError(result.error);
        return;
      }
      setItems((current) => current.filter((item) => item.id !== id));
      setError(null);
    });
  }

  function togglePause() {
    const next = !enabled;
    startTransition(async () => {
      const result = await updateSettings({ memoryEnabled: next });
      if (result.ok) setEnabled(next);
      else setError(result.error);
    });
  }

  return (
    <section className={`${styles.card} ${styles.timelineCard}`} aria-labelledby="memory-title">
      <div className={styles.cardHead}>
        <h2 id="memory-title">Memory timeline</h2>
        <button
          type="button"
          className={`${styles.pauseButton} ${enabled ? "" : styles.paused}`}
          onClick={togglePause}
          disabled={pending}
          aria-pressed={!enabled}
        >
          {enabled ? (
            "Pause"
          ) : (
            <>
              Paused · <span className={styles.resume}>Resume</span>
            </>
          )}
        </button>
      </div>

      {!enabled && (
        <p className={styles.pausedNote}>Kelsie isn&apos;t saving anything new while paused.</p>
      )}

      {items.length === 0 ? (
        <div className={styles.empty}>
          <h3>Nothing saved yet</h3>
          <p>
            As you read and work on the sites you&apos;ve allowed, Kelsie saves short notes here
            so it can bring them back when they&apos;re useful.
          </p>
        </div>
      ) : (
        <ol className={styles.timeline}>
          {items.map((item, index) => (
            <li key={item.id} className={`${styles.note} ${index === 0 ? styles.newest : ""}`}>
              <span className={styles.dot} aria-hidden="true" />
              {editingId === item.id ? (
                <div className={styles.editor}>
                  <label className="visually-hidden" htmlFor={`memory-${item.id}`}>
                    Edit note
                  </label>
                  <textarea
                    id={`memory-${item.id}`}
                    value={draft}
                    maxLength={MEMORY_MAX_LENGTH}
                    rows={3}
                    onChange={(e) => setDraft(e.target.value)}
                    autoFocus
                  />
                  <div className={styles.editorActions}>
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      onClick={() => save(item.id)}
                      disabled={pending}
                    >
                      {pending ? "Saving…" : "Save"}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        setEditingId(null);
                        setError(null);
                      }}
                      disabled={pending}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className={styles.noteBody}>
                  <p>{item.content}</p>
                  <div className={styles.noteFooter}>
                    <span className={styles.meta}>{item.meta}</span>
                    <span className={styles.noteActions}>
                      <button
                        type="button"
                        className={styles.iconButton}
                        onClick={() => startEditing(item)}
                        aria-label="Edit note"
                        disabled={pending}
                      >
                        <PencilIcon />
                      </button>
                      <button
                        type="button"
                        className={styles.iconButton}
                        onClick={() => remove(item.id)}
                        aria-label="Delete note"
                        disabled={pending}
                      >
                        <TrashIcon />
                      </button>
                    </span>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ol>
      )}

      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
      <p className={styles.footnote}>Short notes only, never full pages. Edit or delete anything, anytime.</p>
    </section>
  );
}
