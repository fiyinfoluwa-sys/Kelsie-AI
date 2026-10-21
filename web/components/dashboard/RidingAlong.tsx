"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import Link from "next/link";
import { setSiteAllowed } from "@/app/profile/actions";
import { SITE_DESCRIPTIONS } from "@/lib/content";
import { SITES, type SiteId } from "@/lib/onboarding";
import styles from "./Dashboard.module.css";

const UNDO_MS = 6000;

export default function RidingAlong({ sites: initialSites }: { sites: SiteId[] }) {
  const [sites, setSites] = useState(initialSites);
  const [removed, setRemoved] = useState<SiteId | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  function change(site: SiteId, allowed: boolean) {
    startTransition(async () => {
      const result = await setSiteAllowed(site, allowed);
      if (!result.ok) {
        setError(result.error);
        return;
      }
      setError(null);
      setSites(result.sites);
      window.clearTimeout(timer.current);
      if (allowed) {
        setRemoved(null);
      } else {
        setRemoved(site);
        timer.current = window.setTimeout(() => setRemoved(null), UNDO_MS);
      }
    });
  }

  const label = (id: SiteId) => SITES.find((s) => s.id === id)?.label ?? id;
  const shown = SITES.filter((s) => sites.includes(s.id));

  return (
    <section className={styles.card} aria-labelledby="sites-title">
      <div className={styles.cardHead}>
        <h2 id="sites-title">Where Kelsie&apos;s riding along</h2>
      </div>

      {shown.length === 0 ? (
        <div className={styles.offEverywhere}>
          <p>
            Kelsie&apos;s off everywhere right now, so it won&apos;t notice or save anything.{" "}
            <Link href="/settings#sites">Turn sites back on in Settings</Link>.
          </p>
        </div>
      ) : (
        <ul className={styles.sites}>
          {shown.map((site) => (
            <li key={site.id} className={styles.site}>
              <div>
                <p className={styles.siteName}>{site.label}</p>
                <p className={styles.meta}>{SITE_DESCRIPTIONS[site.id]}</p>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => change(site.id, false)}
                disabled={pending}
                aria-label={`Remove ${site.label}`}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className={styles.toastSlot} role="status">
        {removed && (
          <p className={styles.toast}>
            Removed {label(removed)}.{" "}
            <button type="button" onClick={() => change(removed, true)} disabled={pending}>
              Undo
            </button>
          </p>
        )}
      </div>
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}

      <Link href="/settings#sites" className={styles.cardLink}>
        Add or manage sites in Settings
      </Link>
    </section>
  );
}
