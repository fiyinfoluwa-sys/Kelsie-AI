"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Orb from "./Orb";
import SpeechBubble from "./SpeechBubble";
import SplitScreen from "./SplitScreen";
import { CheckIcon } from "./icons";
import {
  ROLES,
  SITES,
  VIBES,
  displayName,
  loadAnswers,
  saveAnswers,
  toggleRole,
  vibeFarewell,
  vibeSample,
  type OnboardingAnswers,
  type Role,
} from "@/lib/onboarding";
import {
  detectTimezone,
  formatTimezone,
  localTime,
  timezoneGroups,
} from "@/lib/timezones";
import styles from "./Onboarding.module.css";

const TOTAL_STEPS = 5;
const DONE = 6;

const STEP_COPY: Record<number, { title: string; subtitle: string }> = {
  1: {
    title: "What should Kelsie call you?",
    subtitle: "First name, nickname, whatever feels right.",
  },
  2: {
    title: "What are you up to?",
    subtitle: "Pick all that apply. This helps Kelsie know what to look out for.",
  },
  3: {
    title: "Is this your timezone?",
    subtitle: "Kelsie uses it so deadlines land at the right time.",
  },
  4: {
    title: "Pick Kelsie's vibe.",
    subtitle: "Tap one to hear the difference. You can change it anytime.",
  },
  5: {
    title: "Where can Kelsie help?",
    subtitle: "Kelsie only runs on the sites you turn on.",
  },
};

const ROLE_REACTIONS: Record<Role, (name: string) => string> = {
  student: (name) =>
    `classes, got it, ${name}. I'll keep an eye on due dates for you.`,
  working: (name) =>
    `work mode, ${name}. I'll help keep your meetings and projects straight.`,
  job_hunting: (name) =>
    `ooh, the hunt is on. I'll help you keep every application on track, ${name}.`,
  mix: (name) => `a bit of everything, ${name}? same. I'll keep up.`,
};

/** How Kelsie refers to each role when several are picked. */
const ROLE_PHRASES: Record<Exclude<Role, "mix">, string> = {
  student: "school",
  working: "work",
  job_hunting: "the job hunt",
};

function roleMessage(roles: Role[], name: string): string {
  if (roles.length === 0) {
    return `nice to meet you, ${name}! so what's keeping you busy these days?`;
  }
  if (roles.length === 1) return ROLE_REACTIONS[roles[0]](name);

  const phrases = roles.map((r) => ROLE_PHRASES[r as Exclude<Role, "mix">]);
  const list =
    phrases.length === 2
      ? phrases.join(" and ")
      : `${phrases.slice(0, -1).join(", ")} and ${phrases[phrases.length - 1]}`;
  return `${list}? that's a lot, ${name}. I'll help keep it all straight.`;
}

function kelsieMessage(step: number, answers: OnboardingAnswers): string {
  const name = displayName(answers);
  switch (step) {
    case 1:
      return answers.name.trim()
        ? `${answers.name.trim()}. love that. nice to meet you!`
        : "hey, I'm Kelsie. first things first, what should I call you?";
    case 2:
      return roleMessage(answers.roles, name);
    case 3:
      return `quick check, ${name}. I'll use your timezone so I never nudge you at 3am.`;
    case 4:
      return answers.vibe
        ? vibeSample(answers.vibe, name)
        : `how should I talk to you, ${name}? tap one to hear it.`;
    case 5:
      return `last one, ${name}. you pick where I show up, and I stay off everywhere else.`;
    default:
      return vibeFarewell(answers.vibe, name);
  }
}

export default function Onboarding() {
  const [answers, setAnswers] = useState<OnboardingAnswers | null>(null);
  const [step, setStep] = useState(1);
  const [now, setNow] = useState(() => new Date());
  const [notice, setNotice] = useState<string | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const firstRender = useRef(true);

  // Saved answers and the timezone only exist in the browser, so load them
  // after the first render instead of during server rendering.
  useEffect(() => {
    const saved = loadAnswers();
    setAnswers({ ...saved, timezone: saved.timezone || detectTimezone() });
    if (saved.completedAt) setStep(DONE);
  }, []);

  useEffect(() => {
    if (answers) saveAnswers(answers);
  }, [answers]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  const ready = answers !== null;

  // Move focus to the new heading so keyboard and screen reader users
  // land on the next question.
  useEffect(() => {
    if (!ready) return;
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    headingRef.current?.focus();
  }, [step, ready]);

  // Built once: the list of zones never changes, and the detected zone
  // only needs to be included if the browser's list is missing it.
  const initialZone = answers?.timezone ?? "";
  const zoneGroups = useMemo(
    () => (ready ? timezoneGroups(initialZone) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ready],
  );

  if (!answers) {
    return (
      <SplitScreen panel={<Orb size={180} />} panelLabel="Kelsie" align="top">
        {null}
      </SplitScreen>
    );
  }

  const name = displayName(answers);

  function update(changes: Partial<OnboardingAnswers>) {
    setAnswers((current) => (current ? { ...current, ...changes } : current));
  }

  function goTo(next: number) {
    setNotice(null);
    setStep(next);
    window.scrollTo({ top: 0 });
  }

  function next(event: React.FormEvent) {
    event.preventDefault();
    if (!canContinue) return;
    if (step === TOTAL_STEPS) {
      update({ completedAt: new Date().toISOString() });
      goTo(DONE);
    } else {
      goTo(step + 1);
    }
  }

  const canContinue =
    (step !== 2 || answers.roles.length > 0) &&
    (step !== 4 || answers.vibe !== null);

  const continueLabel =
    step === 3 ? "Yes, that's right" : step === TOTAL_STEPS ? "Finish setup" : "Continue";

  const header =
    step === DONE ? (
      <Progress label="All done" value={1} />
    ) : (
      <Progress label={`Step ${step} of ${TOTAL_STEPS}`} value={step / TOTAL_STEPS} />
    );

  const panel = (
    <div className={styles.panelArt}>
      <Orb size={180} />
      <SpeechBubble tone="dark" tail="bottom" className={styles.bubble} live>
        <span key={`${step}-${answers.roles.join()}-${answers.vibe}`} className={styles.message}>
          {kelsieMessage(step, answers)}
        </span>
      </SpeechBubble>
    </div>
  );

  if (step === DONE) {
    const roles = ROLES.filter((r) => answers.roles.includes(r.id)).map((r) => r.label);
    const vibe = VIBES.find((v) => v.id === answers.vibe);
    const sites = SITES.filter((s) => answers.sites[s.id]).map((s) => s.label);

    return (
      <SplitScreen panel={panel} panelLabel="Kelsie" header={header} align="top">
        <h1 ref={headingRef} tabIndex={-1} className={styles.title}>
          You&apos;re all set, {name}!
        </h1>
        <p className={styles.subtitle}>
          Last thing: add the extension so Kelsie can ride along.
        </p>

        <div className={styles.doneActions}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() =>
              setNotice("The Chrome extension isn't in the Web Store yet. Coming soon!")
            }
          >
            Add Kelsie to Chrome
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => setNotice("Your profile page is coming next.")}
          >
            Go to my profile
          </button>
        </div>
        <p className={styles.notice} role="status">
          {notice}
        </p>

        <dl className={styles.summary}>
          <SummaryRow label="Name" value={name} onChange={() => goTo(1)} />
          <SummaryRow
            label="Up to"
            value={roles.length ? roles.join(", ") : "Not set"}
            onChange={() => goTo(2)}
          />
          <SummaryRow
            label="Timezone"
            value={formatTimezone(answers.timezone)}
            onChange={() => goTo(3)}
          />
          <SummaryRow label="Vibe" value={vibe?.label ?? "Not set"} onChange={() => goTo(4)} />
          <SummaryRow
            label="Sites"
            value={sites.length ? sites.join(", ") : "None yet"}
            onChange={() => goTo(5)}
          />
        </dl>
      </SplitScreen>
    );
  }

  const copy = STEP_COPY[step];

  return (
    <SplitScreen panel={panel} panelLabel="Kelsie" header={header} align="top">
      <form onSubmit={next} className={styles.form}>
        <h1 ref={headingRef} tabIndex={-1} className={styles.title}>
          {copy.title}
        </h1>
        <p className={styles.subtitle}>{copy.subtitle}</p>

        {step === 1 && (
          <label className={styles.field}>
            <span className="visually-hidden">Your name</span>
            <input
              type="text"
              name="name"
              autoComplete="given-name"
              placeholder="Your name"
              maxLength={40}
              value={answers.name}
              onChange={(e) => update({ name: e.target.value })}
              autoFocus
            />
          </label>
        )}

        {step === 2 && (
          <fieldset className={styles.cards}>
            <legend className="visually-hidden">
              What are you up to? Pick all that apply.
            </legend>
            {ROLES.map((role) => (
              <label key={role.id} className={styles.card}>
                <input
                  type="checkbox"
                  name="roles"
                  value={role.id}
                  checked={answers.roles.includes(role.id)}
                  onChange={() => update({ roles: toggleRole(answers.roles, role.id) })}
                />
                <span className={styles.cardCheck} aria-hidden="true">
                  <CheckIcon />
                </span>
                <span className={styles.cardTitle}>{role.label}</span>
                <span className={styles.cardDetail}>{role.detail}</span>
              </label>
            ))}
          </fieldset>
        )}

        {step === 3 && (
          <div className={styles.timezone}>
            <div className={styles.zoneCard}>
              <span className={styles.zoneName}>{formatTimezone(answers.timezone)}</span>
              <span className={styles.zoneTime}>
                It&apos;s {localTime(answers.timezone, now)} there right now
              </span>
            </div>
            <label className={styles.field}>
              <span>Not right? Pick yours</span>
              <select
                value={answers.timezone}
                onChange={(e) => update({ timezone: e.target.value })}
              >
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
          </div>
        )}

        {step === 4 && (
          <fieldset className={`${styles.cards} ${styles.cardsStacked}`}>
            <legend className="visually-hidden">Kelsie's vibe</legend>
            {VIBES.map((vibe) => (
              <label key={vibe.id} className={styles.card}>
                <input
                  type="radio"
                  name="vibe"
                  value={vibe.id}
                  checked={answers.vibe === vibe.id}
                  onChange={() => update({ vibe: vibe.id })}
                />
                <span className={styles.cardTitle}>{vibe.label}</span>
                <span className={styles.cardDetail}>“{vibeSample(vibe.id, name)}”</span>
              </label>
            ))}
          </fieldset>
        )}

        {step === 5 && (
          <>
            <fieldset className={styles.sites}>
              <legend className="visually-hidden">Sites Kelsie can run on</legend>
              {SITES.map((site) => (
                <label key={site.id} className={styles.site}>
                  <input
                    type="checkbox"
                    name={site.id}
                    checked={answers.sites[site.id]}
                    onChange={(e) =>
                      update({ sites: { ...answers.sites, [site.id]: e.target.checked } })
                    }
                  />
                  <span className={styles.checkbox} aria-hidden="true">
                    <CheckIcon />
                  </span>
                  {site.label}
                </label>
              ))}
            </fieldset>

            <div className={styles.privacyNote}>
              <h2>What Kelsie keeps</h2>
              <ul>
                <li>Short notes about pages, never full copies.</li>
                <li>Never reads password or payment fields.</li>
                <li>You can see and delete everything on your profile.</li>
              </ul>
            </div>
          </>
        )}

        <div className={styles.nav}>
          {step > 1 && (
            <button type="button" className="btn btn-secondary" onClick={() => goTo(step - 1)}>
              Back
            </button>
          )}
          <button
            type="submit"
            className={`btn btn-primary ${styles.continue}`}
            disabled={!canContinue}
          >
            {continueLabel}
          </button>
        </div>
      </form>
    </SplitScreen>
  );
}

function Progress({ label, value }: { label: string; value: number }) {
  return (
    <div className={styles.progress}>
      <span className={styles.progressLabel}>{label}</span>
      <div
        className={styles.progressTrack}
        role="progressbar"
        aria-label="Setup progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(value * 100)}
      >
        <div className={styles.progressFill} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

function SummaryRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: () => void;
}) {
  return (
    <div className={styles.summaryRow}>
      <dt>{label}</dt>
      <dd>{value}</dd>
      <button type="button" onClick={onChange} aria-label={`Change ${label.toLowerCase()}`}>
        Change
      </button>
    </div>
  );
}
