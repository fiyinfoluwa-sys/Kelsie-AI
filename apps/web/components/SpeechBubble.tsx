import styles from "./SpeechBubble.module.css";

type SpeechBubbleProps = {
  children: React.ReactNode;
  /** Which side of the bubble the tail points from. */
  tail?: "left" | "bottom";
  tone?: "light" | "dark";
  className?: string;
  live?: boolean;
};

export default function SpeechBubble({
  children,
  tail = "left",
  tone = "light",
  className,
  live = false,
}: SpeechBubbleProps) {
  return (
    <div
      className={`${styles.bubble} ${styles[tail]} ${styles[tone]} ${className ?? ""}`}
      aria-live={live ? "polite" : undefined}
    >
      {children}
    </div>
  );
}
