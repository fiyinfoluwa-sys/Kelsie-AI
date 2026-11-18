import Link from "next/link";
import Logo from "./Logo";
import styles from "./Nav.module.css";

export default function Nav() {
  return (
    <header className={styles.header}>
      <nav className={`container ${styles.nav}`} aria-label="Main">
        <Logo />
        <ul className={styles.links}>
          <li>
            <a href="#features">Features</a>
          </li>
          <li>
            <a href="#vibes">Vibes</a>
          </li>
          <li>
            <a href="#privacy">Privacy</a>
          </li>
        </ul>
        <div className={styles.actions}>
          <Link href="/signin" className={styles.signIn}>
            Sign in
          </Link>
          <Link href="/signup" className="btn btn-primary btn-sm">
            Get Kelsie
          </Link>
        </div>
      </nav>
    </header>
  );
}
