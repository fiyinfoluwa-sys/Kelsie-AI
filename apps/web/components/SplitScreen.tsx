import Logo from "./Logo";
import styles from "./SplitScreen.module.css";

type SplitScreenProps = {
  /** Main content on the left. */
  children: React.ReactNode;
  /** Content of the dark panel on the right (below on phones). */
  panel: React.ReactNode;
  panelLabel: string;
  /** Shown above the content, in the same column, such as a progress bar. */
  header?: React.ReactNode;
  /**
   * "center" vertically centers the content; "top" pins it at a fixed
   * height so it doesn't jump around when its length changes.
   */
  align?: "center" | "top";
};

export default function SplitScreen({
  children,
  panel,
  panelLabel,
  header,
  align = "center",
}: SplitScreenProps) {
  return (
    <div className={styles.page}>
      <main className={styles.content}>
        <div className={styles.top}>
          <Logo />
        </div>
        <div
          className={`${styles.wrap} ${align === "top" ? styles.alignTop : styles.alignCenter}`}
        >
          {header && <div className={styles.header}>{header}</div>}
          {children}
        </div>
      </main>

      <aside className={styles.panel} aria-label={panelLabel}>
        <div className={styles.panelInner}>{panel}</div>
      </aside>
    </div>
  );
}
