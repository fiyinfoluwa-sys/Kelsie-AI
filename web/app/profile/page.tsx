import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import DeleteAccount from "@/components/DeleteAccount";
import Logo from "@/components/Logo";
import Orb from "@/components/Orb";
import SpeechBubble from "@/components/SpeechBubble";
import { signOut } from "@/app/auth/actions";
import { ROLES, SITES, VIBES, displayName, rowToAnswers, vibeFarewell } from "@/lib/onboarding";
import { getProfile, requireUser } from "@/lib/session";
import { formatTimezone } from "@/lib/timezones";
import styles from "@/components/Profile.module.css";

export const metadata: Metadata = {
  title: "Your profile — Kelsie",
};

export default async function ProfilePage() {
  const { supabase, user } = await requireUser();
  const profile = await getProfile(supabase, user.id);
  if (!profile?.onboarding_completed_at) redirect("/onboarding");

  const answers = rowToAnswers(profile);
  const name = displayName(answers);
  const rows = [
    { label: "Name", value: name },
    {
      label: "Up to",
      value: ROLES.filter((r) => answers.roles.includes(r.id)).map((r) => r.label).join(", "),
    },
    { label: "Timezone", value: formatTimezone(answers.timezone) },
    { label: "Vibe", value: VIBES.find((v) => v.id === answers.vibe)?.label ?? "Not set" },
    {
      label: "Sites",
      value: SITES.filter((s) => answers.sites[s.id]).map((s) => s.label).join(", ") || "None yet",
    },
  ];

  return (
    <div className={styles.page}>
      <header className={`container ${styles.header}`}>
        <Logo />
        <form action={signOut}>
          <button type="submit" className="btn btn-secondary btn-sm">
            Sign out
          </button>
        </form>
      </header>

      <main className={`container ${styles.main}`}>
        <section className={styles.hello}>
          <Orb size={96} />
          <div>
            <h1 className={styles.title}>Hey, {name}.</h1>
            <SpeechBubble className={styles.bubble}>{vibeFarewell(answers.vibe, name)}</SpeechBubble>
          </div>
        </section>

        <section className={styles.card} aria-labelledby="setup-title">
          <div className={styles.cardHead}>
            <h2 id="setup-title">Your setup</h2>
            <Link href="/onboarding" className={styles.edit}>
              Edit
            </Link>
          </div>
          <dl className={styles.rows}>
            {rows.map((row) => (
              <div key={row.label} className={styles.row}>
                <dt>{row.label}</dt>
                <dd>{row.value}</dd>
              </div>
            ))}
          </dl>
          <p className={styles.account}>Signed in as {user.email}</p>
        </section>

        <section className={`${styles.card} ${styles.dangerZone}`} aria-labelledby="danger-title">
          <h2 id="danger-title">Delete account</h2>
          <p className={styles.muted}>
            Removes your account and all of your Kelsie data for good.
          </p>
          <DeleteAccount />
        </section>
      </main>
    </div>
  );
}
