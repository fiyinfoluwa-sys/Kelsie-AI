import Link from "next/link";
import Nav from "@/components/Nav";
import Orb from "@/components/Orb";
import SpeechBubble from "@/components/SpeechBubble";
import VibePicker from "@/components/VibePicker";
import Logo from "@/components/Logo";
import {
  ApplicationsIcon,
  CheckIcon,
  FocusIcon,
  MemoryIcon,
  RadarIcon,
} from "@/components/icons";
import styles from "./page.module.css";

const FEATURES = [
  {
    title: "Remembers for you",
    body: "Read an article, open a Google Doc, and Kelsie already has the quote you needed. Short summaries only, never full page copies.",
    Icon: MemoryIcon,
  },
  {
    title: "Deadline radar",
    body: "Canvas, syllabus PDFs, emails, job posts. Kelsie spots the due dates and lines them up in your timezone. You confirm before anything's saved.",
    Icon: RadarIcon,
  },
  {
    title: "Focus mode",
    body: "Bouncing between 14 tabs? Kelsie offers a gentle “want to lock in?” with a timer and just the tabs that matter. Snooze it anytime.",
    Icon: FocusIcon,
  },
  {
    title: "Applications handled",
    body: "Autofills job and internship forms from your profile, drafts a cover letter for the posting, and keeps track of everywhere you've applied.",
    Icon: ApplicationsIcon,
  },
];

const PRIVACY_POINTS = [
  {
    title: "Only where you say so",
    body: "You pick the sites Kelsie runs on. Everywhere else, it's off.",
  },
  {
    title: "Summaries, not copies",
    body: "Kelsie saves short notes about pages, not the pages themselves.",
  },
  {
    title: "Hands off the sensitive stuff",
    body: "Password and payment fields are never read. Ever.",
  },
  {
    title: "Your memory, your call",
    body: "See everything Kelsie remembers on your profile. Edit or delete any of it.",
  },
];

export default function Home() {
  return (
    <>
      <Nav />

      <main>
        <section className={`container ${styles.hero}`}>
          <div className={styles.heroCopy}>
            <span className="eyebrow">Your ambient AI sidekick</span>
            <h1 className={styles.heroTitle}>Meet Kelsie.</h1>
            <p className={styles.heroLede}>
              Kelsie rides along in your browser, picks up on what you&apos;re
              doing, and steps in when it matters. No prompting required.
            </p>
            <div className={styles.heroActions}>
              <Link href="/signup" className="btn btn-primary">
                Get started, it&apos;s free
              </Link>
              <a href="#features" className="btn btn-secondary">
                See what it does
              </a>
            </div>
            <p className={styles.heroNote}>
              Works as a Chrome extension on the sites you choose.
            </p>
          </div>

          <div className={styles.heroArt}>
            <Orb size={240} className={styles.heroOrb} />
            <SpeechBubble className={styles.heroBubble} tail="bottom">
              hey, I&apos;m Kelsie. I&apos;ll remember the stuff you&apos;d
              rather not.
            </SpeechBubble>
          </div>
        </section>

        <section id="features" className={`container ${styles.section}`}>
          <div className={styles.sectionHead}>
            <span className="eyebrow">What Kelsie does</span>
            <h2 className={styles.sectionTitle}>
              Help that shows up before you ask.
            </h2>
          </div>
          <ul className={styles.features}>
            {FEATURES.map(({ title, body, Icon }) => (
              <li key={title} className={styles.card}>
                <span className={styles.cardIcon}>
                  <Icon />
                </span>
                <h3 className={styles.cardTitle}>{title}</h3>
                <p className={styles.cardBody}>{body}</p>
              </li>
            ))}
          </ul>
        </section>

        <section id="vibes" className={`container ${styles.section}`}>
          <div className={styles.vibes}>
            <div className={styles.sectionHead}>
              <span className="eyebrow">Pick a vibe</span>
              <h2 className={styles.sectionTitle}>
                Same Kelsie. Your kind of energy.
              </h2>
              <p className={styles.sectionLede}>
                Choose how Kelsie talks to you. Tap one to hear the difference.
              </p>
            </div>
            <VibePicker />
          </div>
        </section>

        <section id="privacy" className={`container ${styles.section}`}>
          <div className={styles.privacy}>
            <div className={styles.sectionHead}>
              <span className="eyebrow">Privacy</span>
              <h2 className={styles.sectionTitle}>
                You&apos;re in control. Always.
              </h2>
              <p className={styles.sectionLede}>
                Kelsie is upfront about what it sees and what it keeps. No
                surprises.
              </p>
            </div>
            <ul className={styles.privacyList}>
              {PRIVACY_POINTS.map((point) => (
                <li key={point.title} className={styles.privacyItem}>
                  <span className={styles.check}>
                    <CheckIcon />
                  </span>
                  <div>
                    <h3 className={styles.privacyTitle}>{point.title}</h3>
                    <p className={styles.privacyBody}>{point.body}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className={styles.signup} aria-labelledby="signup-title">
          <div className={`container ${styles.signupInner}`}>
            <Orb size={120} />
            <h2 id="signup-title" className={styles.signupTitle}>
              Your browser just got a sidekick.
            </h2>
            <p className={styles.signupLede}>
              Setup takes under a minute. Kelsie handles the rest.
            </p>
            <Link href="/signup" className="btn btn-primary">
              Create your account
            </Link>
          </div>
        </section>
      </main>

      <footer className={styles.footer}>
        <div className={`container ${styles.footerBar}`}>
          <Logo tone="dark" />
          <p>© {new Date().getFullYear()} Kelsie</p>
        </div>
      </footer>
    </>
  );
}
