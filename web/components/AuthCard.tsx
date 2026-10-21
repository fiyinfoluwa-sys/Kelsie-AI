import Orb from "./Orb";
import SpeechBubble from "./SpeechBubble";
import SplitScreen from "./SplitScreen";
import styles from "./AuthPage.module.css";

/** Layout for the smaller account pages (forgot / reset password). */
export default function AuthCard({
  bubble,
  children,
}: {
  bubble: string;
  children: React.ReactNode;
}) {
  const panel = (
    <div className={styles.panelArt}>
      <Orb size={150} />
      <SpeechBubble tone="dark" className={styles.panelBubble}>
        {bubble}
      </SpeechBubble>
    </div>
  );
  return (
    <SplitScreen panel={panel} panelLabel="Kelsie">
      {children}
    </SplitScreen>
  );
}
