"""Evaluate the classifier and pick the confidence threshold.

1. Threshold is chosen on VALIDATION only: the lowest confidence where messages
   the router keeps local/specific are >= TARGET_PRECISION correct.
2. Reports on the held-out-template TEST set and the HANDWRITTEN set:
   raw accuracy, macro-F1, per-intent report, confusion matrix, and routing
   stats (coverage = share not sent to the fallback; accuracy of the rest).

Usage: python evaluate.py [--target 0.95]
Writes artifacts/threshold.json and artifacts/eval_report.json
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from common import ARTIFACTS, ROOT, load_jsonl, load_model, to_tensors
from intents import FALLBACK, LABEL2ID, LABELS


@torch.no_grad()
def calibrated_probs(model, x, temperature):
    return torch.softmax(model(x) / temperature, dim=-1).numpy()


def choose_threshold(probs, y, target):
    conf = probs.max(1)
    pred = probs.argmax(1)
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        keep = conf >= t
        if keep.sum() == 0:
            break
        if (pred[keep] == y[keep]).mean() >= target:
            return float(t)
    return 1.0


def routing_stats(probs, y, threshold):
    conf, pred = probs.max(1), probs.argmax(1)
    routed = np.where(conf >= threshold, pred, LABEL2ID[FALLBACK])
    kept = conf >= threshold
    # a "wrong action" = sent somewhere other than the fallback AND wrong
    wrong_action = (routed != y) & (routed != LABEL2ID[FALLBACK])
    return {
        "coverage": float(kept.mean()),
        "accuracy_when_kept": float((pred[kept] == y[kept]).mean()) if kept.any() else None,
        "fell_back_share": float((~kept).mean()),
        "wrong_action_rate": float(wrong_action.mean()),
        "end_to_end_ok": float(((routed == y) | (routed == LABEL2ID[FALLBACK])).mean()),
    }


def report(name, model, temperature, threshold, rows):
    x, y = to_tensors(rows)
    y = y.numpy()
    probs = calibrated_probs(model, x, temperature)
    pred = probs.argmax(1)
    out = {
        "n": len(rows),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_intent": classification_report(y, pred, target_names=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(LABELS))).tolist(),
        "routing": routing_stats(probs, y, threshold),
        "errors": [{"text": r["text"], "label": r["label"], "pred": LABELS[p], "conf": round(float(c), 3)}
                   for r, p, c, t in zip(rows, pred, probs.max(1), y) if p != t],
    }
    print(f"\n=== {name} (n={out['n']}) ===")
    print(f"accuracy {out['accuracy']:.3f}   macro-F1 {out['macro_f1']:.3f}")
    print(classification_report(y, pred, target_names=LABELS, digits=3, zero_division=0))
    r = out["routing"]
    print(f"routing @ threshold {threshold:.2f}: coverage {r['coverage']:.1%}, accuracy of kept "
          f"{r['accuracy_when_kept']:.1%}, sent to fallback {r['fell_back_share']:.1%}, "
          f"wrong action {r['wrong_action_rate']:.1%}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--artifacts", default=str(ARTIFACTS))
    args = ap.parse_args()
    from pathlib import Path
    data_dir, art = Path(args.data_dir), Path(args.artifacts)
    model, ckpt = load_model(art / "intent_fp32.pt")
    T = ckpt["temperature"]

    xv, yv = to_tensors(load_jsonl(data_dir / "val.jsonl"))
    threshold = choose_threshold(calibrated_probs(model, xv, T), yv.numpy(), args.target)
    (art / "threshold.json").write_text(json.dumps(
        {"threshold": threshold, "target_precision": args.target, "temperature": T}, indent=2))
    print(f"threshold chosen on validation: {threshold:.2f} (target {args.target:.0%} accuracy on kept)")
    # everything a non-Python runtime (js/router.js) needs besides the .onnx file
    from intents import ROUTE
    (art / "router_config.json").write_text(json.dumps(
        {"labels": LABELS, "routes": ROUTE, "fallback": FALLBACK, "temperature": T,
         "threshold": threshold}, indent=2))

    results = {"threshold": threshold, "temperature": T}
    for name, path in [("test (held-out templates)", data_dir / "test.jsonl"),
                       ("handwritten", ROOT / "data/handwritten_test.jsonl")]:
        results[name] = report(name, model, T, threshold, load_jsonl(path))
    (art / "eval_report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
