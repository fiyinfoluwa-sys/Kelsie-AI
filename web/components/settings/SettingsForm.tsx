"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { signOut } from "@/app/auth/actions";
import { clearMemories, sendPasswordChangeEmail, updateSettings } from "@/app/settings/actions";
import DeleteAccount from "@/components/DeleteAccount";
import { CheckIcon } from "@/components/icons";
import {
  NAME_MAX_LENGTH,
  ROLES,
  SITES,
  VIBES,
  displayName,
  toggleRole,
  vibeSample,
  type OnboardingAnswers,
  type Role,
  type SiteId,
  type Vibe,
} from "@/lib/onboarding";
import { notifyExtension } from "@/lib/extension";
import { formatTimezone, timezoneGroups } from "@/lib/timezones";
import onboarding from "@/components/Onboarding.module.css";
import styles from "./Settings.module.css";

type Status = { state: "idle" | "saving" | "saved" } | { state: "error"; message: string };
type Section = "about" | "vibe" | "sites" | "memory";

type SettingsFormProps = {
  initial: OnboardingAnswers;
  memoryEnabled: boolean;
  memoryCount: number;
  email: string;
  provider: "email" | "google";
};

export default function SettingsForm({
  initial,
  memoryEnabled: initialMemory,
  memoryCount: initialMemoryCount,
  email,
  provider,
}: SettingsFormProps) {
  const [name, setName] = useState(initial.name);
  const [roles, setRoles] = useState<Role[]>(initial.roles);
  const [timezone, setTimezone] = useState(initial.timezone);
  const [vibe, setVibe] = useState<Vibe | null>(initial.vibe);
  const [sites, setSites] = useState<Record<SiteId, boolean>>(initial.sites);
  const [memoryEnabled, setMemoryEnabled] = useState(initialMemory);
  const [memoryCount, setMemoryCount] = useState(initialMemoryCount);
  const [status, setStatus] = useState<Record<Section, Status>>({
    about: { state: "idle" },
    vibe: { state: "idle" },
    sites: { state: "idle" },
    memory: { state: "idle" },
  });
  const [zoneGroups] = useState(() => timezoneGroups(initial.timezone));
  const savedTimers = useRef<Partial<Record<Section, number>>>({});
  const nameTimer = useRef<number | undefined>(undefined);
  const lastSavedName = useRef(initial.name);

  useEffect(() => {
    const timers = savedTimers.current;
    return () => {
      Object.values(timers).forEach((t) => window.clearTimeout(t));
      window.clearTimeout(nameTimer.current);
    };
  }, []);

  /** Saves one change and shows "Saved" next to that section's title. */
  async function save(section: Section, changes: Parameters<typeof updateSettings>[0]) {
    setStatus((s) => ({ ...s, [section]: { state: "saving" } }));
    const result = await updateSettings(changes);
    window.clearTimeout(savedTimers.current[section]);
    if (result.ok) {
      // Sites, name, vibe and pause all show in the extension: re-sync it.
      notifyExtension();
      setStatus((s) => ({ ...s, [section]: { state: "saved" } }));
      savedTimers.current[section] = window.setTimeout(
        () => setStatus((s) => ({ ...s, [section]: { state: "idle" } })),
        2000,
      );
    } else {
      setStatus((s) => ({ ...s, [section]: { state: "error", message: result.error } }));
    }
    return result.ok;
  }

  // The name saves once you pause typing, or when you leave the field.
  function changeName(value: string) {
    setName(value);
    window.clearTimeout(nameTimer.current);
    nameTimer.current = window.setTimeout(() => flushName(value), 700);
  }
  function flushName(value = name) {
    window.clearTimeout(nameTimer.current);
    if (value === lastSavedName.current) return;
    lastSavedName.current = value;
    void save("about", { name: value });
  }

  function changeRole(role: Role) {
    const next = toggleRole(roles, role);
    if (next.length === 0) {
      setStatus((s) => ({ ...s, about: { state: "error", message: "Pick at least one." } }));
      return;
    }
    const previous = roles;
    setRoles(next);
    void save("about", { roles: next }).then((ok) => !ok && setRoles(previous));
  }

  function changeTimezone(value: string) {
    const previous = timezone;
    setTimezone(value);
    void save("about", { timezone: value }).then((ok) => !ok && setTimezone(previous));
  }

  function changeVibe(value: Vibe) {
    const previous = vibe;
    setVibe(value);
    void save("vibe", { vibe: value }).then((ok) => !ok && setVibe(previous));
  }

  function changeSite(id: SiteId, allowed: boolean) {
    const previous = sites;
    const next = { ...sites, [id]: allowed };
    setSites(next);
    void save("sites", { sites: next }).then((ok) => !ok && setSites(previous));
  }

  function changeMemory(enabled: boolean) {
    setMemoryEnabled(enabled);
    void save("memory", { memoryEnabled: enabled }).then((ok) => !ok && setMemoryEnabled(!enabled));
  }

  const kelsieName = displayName({ name });

  return (
    <main className={`container ${styles.page}`}>
      <h1 className={styles.title}>Settings</h1>
      <p className={styles.subtitle}>Changes save automatically.</p>

      {/* About you */}
      <section className={styles.card} aria-labelledby="about-title">
        <SectionHead id="about-title" title="About you" status={status.about} />

        <label className={styles.field}>
          <span className={styles.label}>Name</span>
          <input
            type="text"
            value={name}
            maxLength={NAME_MAX_LENGTH}
            autoComplete="given-name"
            onChange={(e) => changeName(e.target.value)}
            onBlur={() => flushName()}
          />
        </label>

        <fieldset className={styles.fieldset}>
          <legend className={styles.label}>What you&apos;re up to</legend>
          <div className={styles.pills}>
            {ROLES.map((role) => (
              <button
                key={role.id}
                type="button"
                className={styles.pill}
                aria-pressed={roles.includes(role.id)}
                onClick={() => changeRole(role.id)}
              >
                {role.label}
              </button>
            ))}
          </div>
          <p className={styles.hint}>
            Pick Job hunting or A mix to see your Applications on the home page.
          </p>
        </fieldset>

        <label className={styles.field}>
          <span className={styles.label}>Timezone</span>
          <select value={timezone} onChange={(e) => changeTimezone(e.target.value)}>
            {zoneGroups.map(([region, zones]) => (
              <optgroup key={region} label={region}>
                {zones.map((zone) => (
                  <option key={zone} value={zone}>
                    {formatTimezone(zone)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
      </section>

      {/* Vibe */}
      <section className={styles.card} aria-labelledby="vibe-title">
        <SectionHead id="vibe-title" title="Kelsie's vibe" status={status.vibe} />
        <fieldset className={`${onboarding.cards} ${onboarding.cardsStacked}`}>
          <legend className="visually-hidden">Kelsie&apos;s vibe</legend>
          {VIBES.map((v) => (
            <label key={v.id} className={onboarding.card}>
              <input
                type="radio"
                name="vibe"
                value={v.id}
                checked={vibe === v.id}
                onChange={() => changeVibe(v.id)}
              />
              <span className={onboarding.cardTitle}>{v.label}</span>
              <span className={onboarding.cardDetail}>“{vibeSample(v.id, kelsieName)}”</span>
            </label>
          ))}
        </fieldset>
      </section>

      {/* Sites */}
      <section id="sites" className={styles.card} aria-labelledby="sites-title">
        <SectionHead id="sites-title" title="Where Kelsie runs" status={status.sites} />
        <fieldset className={onboarding.sites}>
          <legend className="visually-hidden">Sites Kelsie can run on</legend>
          {SITES.map((site) => (
            <label key={site.id} className={onboarding.site}>
              <input
                type="checkbox"
                name={site.id}
                checked={sites[site.id]}
                onChange={(e) => changeSite(site.id, e.target.checked)}
              />
              <span className={onboarding.checkbox} aria-hidden="true">
                <CheckIcon />
              </span>
              {site.label}
            </label>
          ))}
        </fieldset>
      </section>

      {/* Memory and privacy */}
      <MemorySection
        enabled={memoryEnabled}
        onChange={changeMemory}
        status={status.memory}
        memoryCount={memoryCount}
        onCleared={() => setMemoryCount(0)}
      />

      {/* Account */}
      <AccountSection email={email} provider={provider} />

      {/* Delete */}
      <section className={`${styles.card} ${styles.danger}`} aria-labelledby="delete-title">
        <h2 id="delete-title">Delete my account and data</h2>
        <p className={styles.muted}>
          Permanently removes your account and everything Kelsie has saved for you.
        </p>
        <DeleteAccount />
      </section>
    </main>
  );
}

function SectionHead({ id, title, status }: { id: string; title: string; status: Status }) {
  return (
    <div className={styles.sectionHead}>
      <h2 id={id}>{title}</h2>
      <span
        className={`${styles.status} ${status.state === "error" ? styles.statusError : ""}`}
        role="status"
      >
        {status.state === "saving" && "Saving…"}
        {status.state === "saved" && "Saved"}
        {status.state === "error" && status.message}
      </span>
    </div>
  );
}

function MemorySection({
  enabled,
  onChange,
  status,
  memoryCount,
  onCleared,
}: {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  status: Status;
  memoryCount: number;
  onCleared: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  function clear() {
    startTransition(async () => {
      const result = await clearMemories();
      if (result.ok) {
        onCleared();
        setConfirming(false);
        setMessage("All memories cleared.");
      } else {
        setMessage(result.error);
      }
    });
  }

  return (
    <section className={styles.card} aria-labelledby="memory-title">
      <SectionHead id="memory-title" title="Memory and privacy" status={status} />

      <div className={styles.row}>
        <div>
          <p className={styles.rowTitle} id="memory-switch-label">
            Let Kelsie remember things
          </p>
          <p className={styles.muted}>
            Short notes from the sites you allow. Turning this off pauses new notes.
          </p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={enabled}
          aria-labelledby="memory-switch-label"
          className={styles.switch}
          onClick={() => onChange(!enabled)}
        >
          <span className={styles.knob} />
        </button>
      </div>

      <div className={styles.row}>
        <div>
          <p className={styles.rowTitle}>Download my data</p>
          <p className={styles.muted}>Everything Kelsie has saved for you, as a JSON file.</p>
        </div>
        <a href="/settings/export" download className="btn btn-secondary btn-sm">
          Download
        </a>
      </div>

      <div className={styles.row}>
        <div>
          <p className={styles.rowTitle}>Clear all memories</p>
          <p className={styles.muted}>
            {memoryCount === 0
              ? "Nothing saved right now."
              : `Deletes all ${memoryCount} note${memoryCount === 1 ? "" : "s"} Kelsie saved. Can't be undone.`}
          </p>
        </div>
        {!confirming ? (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              setConfirming(true);
              setMessage(null);
            }}
            disabled={memoryCount === 0}
          >
            Clear
          </button>
        ) : (
          <div className={styles.confirm}>
            <button type="button" className={`btn btn-sm ${styles.dangerButton}`} onClick={clear} disabled={pending}>
              {pending ? "Clearing…" : "Yes, clear them"}
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setConfirming(false)}
              disabled={pending}
            >
              Cancel
            </button>
          </div>
        )}
      </div>
      <p className={styles.note} role="status">
        {message}
      </p>
    </section>
  );
}

function AccountSection({ email, provider }: { email: string; provider: "email" | "google" }) {
  const [message, setMessage] = useState<string | null>(null);
  const [pending, startTransition] = useTransition();

  function changePassword() {
    startTransition(async () => {
      const result = await sendPasswordChangeEmail();
      setMessage(
        result.ok
          ? `We sent a link to ${email}. Open it in this browser to choose a new password.`
          : result.error,
      );
    });
  }

  return (
    <section className={styles.card} aria-labelledby="account-title">
      <div className={styles.sectionHead}>
        <h2 id="account-title">Account</h2>
      </div>

      <div className={styles.row}>
        <div>
          <p className={styles.rowTitle}>
            Signed in with {provider === "google" ? "Google" : "email and password"}
          </p>
          <p className={styles.muted}>{email}</p>
        </div>
      </div>

      {provider === "email" && (
        <div className={styles.row}>
          <div>
            <p className={styles.rowTitle}>Change password</p>
            <p className={styles.muted}>We&apos;ll email you a secure link to set a new one.</p>
          </div>
          <button type="button" className="btn btn-secondary btn-sm" onClick={changePassword} disabled={pending}>
            {pending ? "Sending…" : "Change password"}
          </button>
        </div>
      )}

      <div className={styles.row}>
        <div>
          <p className={styles.rowTitle}>Sign out</p>
          <p className={styles.muted}>Signs you out on this browser.</p>
        </div>
        <form action={signOut}>
          <button type="submit" className="btn btn-secondary btn-sm">
            Sign out
          </button>
        </form>
      </div>
      <p className={styles.note} role="status">
        {message}
      </p>
    </section>
  );
}
