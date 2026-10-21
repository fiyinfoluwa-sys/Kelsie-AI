import Orb from "@/components/Orb";
import QuickActions from "./QuickActions";
import styles from "./Dashboard.module.css";

type OrbHeroProps = {
  connected: boolean;
  siteCount: number;
  date: string;
  title: string;
  line: string;
};

export default function OrbHero({ connected, siteCount, date, title, line }: OrbHeroProps) {
  const status = !connected
    ? { live: false, text: "Not in Chrome yet" }
    : siteCount === 0
      ? { live: false, text: "Off on every site" }
      : { live: true, text: `Riding along on ${siteCount} site${siteCount === 1 ? "" : "s"}` };

  return (
    <section className={styles.hero} aria-label="Kelsie">
      <p className={styles.statusPill}>
        <span className={`${styles.statusDot} ${status.live ? styles.live : ""}`} aria-hidden="true" />
        {status.text}
      </p>

      <div className={styles.orbStage}>
        <span className={styles.ringOuter} aria-hidden="true" />
        <span className={styles.ringInner} aria-hidden="true" />
        <Orb size={190} />
      </div>

      <p className={styles.date}>{date}</p>
      <h1 className={styles.greeting}>{title}</h1>
      <p className={styles.greetingLine}>{line}</p>

      <QuickActions connected={connected} />
    </section>
  );
}
