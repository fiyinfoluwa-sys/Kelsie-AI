"""How stable are the results? Re-run the full pipeline (new template split + new
training seed) several times and report mean +/- std.

Usage: python experiments/seed_sweep.py --seeds 1 2 3 4 5
Writes artifacts/seed_sweep.json
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    keys = [("test (held-out templates)", "accuracy"), ("test (held-out templates)", "macro_f1"),
            ("handwritten", "accuracy"), ("handwritten", "macro_f1")]
    rkeys = ["coverage", "accuracy_when_kept", "wrong_action_rate"]
    runs = []
    for seed in args.seeds:
        with tempfile.TemporaryDirectory() as tmp:
            data, art = Path(tmp) / "data", Path(tmp) / "art"
            run(["data/generate.py", "--out", str(data), "--per-template", "10", "--seed", str(seed)])
            run(["train.py", "--data-dir", str(data), "--out-dir", str(art), "--seed", str(seed)])
            run(["evaluate.py", "--data-dir", str(data), "--artifacts", str(art)])
            rep = json.loads((art / "eval_report.json").read_text())
        row = {"seed": seed, "threshold": rep["threshold"]}
        for split, k in keys:
            row[f"{split}/{k}"] = rep[split][k]
        for split in ["test (held-out templates)", "handwritten"]:
            for k in rkeys:
                row[f"{split}/routing/{k}"] = rep[split]["routing"][k]
        runs.append(row)
        print(f"seed {seed}: test acc {row['test (held-out templates)/accuracy']:.3f}  "
              f"handwritten acc {row['handwritten/accuracy']:.3f}  threshold {row['threshold']:.2f}")
    summary = {}
    for k in runs[0]:
        if k == "seed":
            continue
        vals = [r[k] for r in runs]
        summary[k] = {"mean": statistics.fmean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    for k, v in summary.items():
        print(f"{k:55s} {v['mean']:.3f} +/- {v['std']:.3f}")
    (ROOT / "artifacts/seed_sweep.json").write_text(json.dumps({"runs": runs, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
