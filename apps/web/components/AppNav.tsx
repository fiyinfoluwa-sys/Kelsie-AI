import Link from "next/link";
import Logo from "./Logo";
import styles from "./AppNav.module.css";

type AppNavProps = {
  active: "home" | "settings";
  name: string;
  showApplications: boolean;
};

/** Top nav for signed-in pages. */
export default function AppNav({ active, name, showApplications }: AppNavProps) {
  const initial = (name.trim()[0] ?? "K").toUpperCase();
  return (
    <header className={styles.header}>
      <nav className={`container ${styles.nav}`} aria-label="Main">
        <span className={styles.brand}>
          <Logo />
        </span>
        <div className={styles.right}>
          <ul className={styles.links}>
            <li>
              <Link href="/profile" aria-current={active === "home" ? "page" : undefined}>
                Home
              </Link>
            </li>
            {showApplications && (
              <li>
                <Link href="/profile#applications">Applications</Link>
              </li>
            )}
            <li>
              <Link href="/settings" aria-current={active === "settings" ? "page" : undefined}>
                Settings
              </Link>
            </li>
          </ul>
          <Link href="/settings" className={styles.avatar} aria-label={`Signed in as ${name}. Open settings`}>
            {initial}
          </Link>
        </div>
      </nav>
    </header>
  );
}
