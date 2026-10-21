"use client";

import { useState } from "react";
import Orb from "./Orb";
import SpeechBubble from "./SpeechBubble";
import styles from "./VibePicker.module.css";

const VIBES = [
  {
    id: "chill",
    label: "Chill",
    message:
      "hey, no stress — you've got 3 things due Thursday. maybe knock out the reading tonight and call it a win?",
  },
  {
    id: "hype",
    label: "Hype",
    message:
      "THURSDAY IS STACKED. 3 things due and you're about to crush every single one. Essay first — let's GO!",
  },
  {
    id: "direct",
    label: "Straight to the point",
    message: "3 due Thursday: essay, lab report, quiz. Start the essay.",
  },
] as const;

type VibeId = (typeof VIBES)[number]["id"];

export default function VibePicker() {
  const [vibe, setVibe] = useState<VibeId>("chill");
  const current = VIBES.find((v) => v.id === vibe) ?? VIBES[0];

  return (
    <div className={styles.picker}>
      <div className={styles.options} role="group" aria-label="Kelsie's vibe">
        {VIBES.map((v) => (
          <button
            key={v.id}
            type="button"
            className={styles.option}
            aria-pressed={v.id === vibe}
            onClick={() => setVibe(v.id)}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div className={styles.stage}>
        <Orb size={120} />
        <SpeechBubble className={styles.bubble} live>
          <span key={current.id} className={styles.message}>
            {current.message}
          </span>
        </SpeechBubble>
      </div>
    </div>
  );
}
