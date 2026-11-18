import type { Metadata } from "next";
import { redirect } from "next/navigation";
import AppNav from "@/components/AppNav";
import ApplicationsTable from "@/components/dashboard/ApplicationsTable";
import ComingUp from "@/components/dashboard/ComingUp";
import MemoryTimeline from "@/components/dashboard/MemoryTimeline";
import OrbHero from "@/components/dashboard/OrbHero";
import RidingAlong from "@/components/dashboard/RidingAlong";
import styles from "@/components/dashboard/Dashboard.module.css";
import { SOURCE_LABELS, greeting, showsApplications } from "@/lib/content";
import { loadDashboard } from "@/lib/dashboard";
import { daysBetween, dateTile, localHour, longDate, relativeDay, shortDate } from "@/lib/dates";
import { displayName, rowToAnswers, sitesToList } from "@/lib/onboarding";
import { getProfile, requireUser } from "@/lib/session";

export const metadata: Metadata = {
  title: "Home — Kelsie",
};

export default async function ProfilePage() {
  const { supabase, user } = await requireUser();
  const profile = await getProfile(supabase, user.id);
  if (!profile?.onboarding_completed_at) redirect("/onboarding");

  const answers = rowToAnswers(profile);
  const name = displayName(answers);
  const tz = answers.timezone;
  const data = await loadDashboard(supabase, user.id, tz);
  const sites = sitesToList(answers.sites);
  const withApplications = showsApplications(answers.roles, data.applications.length);
  const hello = greeting(answers.vibe, name, localHour(tz), data.upcoming.length);

  return (
    <>
      <AppNav active="home" name={name} showApplications={withApplications} />
      <main className={`container ${styles.page}`}>
        <div className={styles.top}>
          <MemoryTimeline
            memoryEnabled={profile.memory_enabled}
            items={data.memories.map((m) => ({
              id: m.id,
              content: m.content,
              meta: `${SOURCE_LABELS[m.source] ?? "Kelsie"} · ${relativeDay(m.created_at, tz)}`,
            }))}
          />
          <OrbHero
            connected={Boolean(profile.extension_connected_at)}
            siteCount={sites.length}
            date={longDate(tz)}
            title={hello.title}
            line={hello.line}
          />
          <ComingUp
            upcoming={data.upcoming.map((d) => {
              const days = daysBetween(data.today, d.due_date);
              return {
                id: d.id,
                title: d.title,
                details: d.details,
                ...dateTile(d.due_date),
                tag: days === 0 ? "Today" : days === 1 ? "Tomorrow" : null,
              };
            })}
            suggested={data.suggested.map((d) => ({
              id: d.id,
              title: d.title,
              details: d.details,
              when: `Due ${shortDate(d.due_date)}`,
            }))}
          />
        </div>

        <div className={styles.below}>
          {withApplications && (
            <ApplicationsTable
              rows={data.applications.map((a) => ({
                id: a.id,
                company: a.company,
                role: a.role,
                status: a.status,
                applied: a.applied_on ? shortDate(a.applied_on) : "—",
              }))}
            />
          )}
          <RidingAlong sites={sites} />
        </div>
      </main>
    </>
  );
}
