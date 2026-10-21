import styles from "./Orb.module.css";

type OrbProps = {
  /** Diameter in pixels. */
  size?: number;
  /** Show the eyes that make the orb read as a character. */
  face?: boolean;
  className?: string;
};

export default function Orb({ size = 220, face = true, className }: OrbProps) {
  return (
    <div
      className={`${styles.wrap} ${className ?? ""}`}
      style={{ "--orb-size": `${size}px` } as React.CSSProperties}
      aria-hidden="true"
    >
      <div className={styles.glow} />
      <div className={styles.orb}>
        {face && (
          <div className={styles.face}>
            <span className={styles.eye} />
            <span className={styles.eye} />
          </div>
        )}
      </div>
    </div>
  );
}
