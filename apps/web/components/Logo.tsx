import Link from "next/link";
import styles from "./Logo.module.css";

export default function Logo({ tone = "light" }: { tone?: "light" | "dark" }) {
  return (
    <Link href="/" className={`${styles.logo} ${styles[tone]}`}>
      <span className={styles.dot} aria-hidden="true" />
      Kelsie
    </Link>
  );
}
