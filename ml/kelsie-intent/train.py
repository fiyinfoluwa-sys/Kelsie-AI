"""Train the intent classifier.

Usage: python train.py [--epochs 40] [--seed 0]
Writes artifacts/intent_fp32.pt (weights + config + temperature) and artifacts/train_log.json.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from common import ARTIFACTS, ROOT, load_jsonl, to_tensors
from intents import LABELS
from model import IntentClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model, x, bs=512):
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])


def fit_temperature(logits, y):
    """Temperature scaling on validation logits so softmax confidences are calibrated."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ARTIFACTS))
    args = ap.parse_args()
    from pathlib import Path
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    set_seed(args.seed)
    torch.set_num_threads(1)

    xtr, ytr = to_tensors(load_jsonl(data_dir / "train.jsonl"))
    xva, yva = to_tensors(load_jsonl(data_dir / "val.jsonl"))

    config = {"dim": 96, "hidden": 128, "dropout": 0.2}
    model = IntentClassifier(num_labels=len(LABELS), **config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1, best_state, bad_epochs, log = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        total = 0.0
        for i in range(0, len(xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = xtr[idx]
            # feature dropout: randomly blank ~10% of n-grams so the model can't lean on one feature
            keep = (torch.rand(xb.shape) > 0.1) | (xb == 0)
            xb = xb * keep
            opt.zero_grad()
            loss = loss_fn(model(xb), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_pred = predict_logits(model, xva).argmax(1)
        val_f1 = f1_score(yva, val_pred, average="macro")
        val_acc = float((val_pred == yva).float().mean())
        log.append({"epoch": epoch, "train_loss": total / len(xtr), "val_acc": val_acc, "val_macro_f1": val_f1})
        print(f"epoch {epoch:2d}  loss {total / len(xtr):.4f}  val acc {val_acc:.4f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1 + 1e-4:
            best_f1, bad_epochs = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    temperature = fit_temperature(predict_logits(model, xva), yva)
    print(f"best val macro-F1 {best_f1:.4f}; temperature {temperature:.3f}; {time.time() - t0:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config, "labels": LABELS,
                "temperature": temperature}, out_dir / "intent_fp32.pt")
    with open(out_dir / "train_log.json", "w") as f:
        json.dump({"args": vars(args), "best_val_macro_f1": best_f1, "temperature": temperature,
                   "epochs": log}, f, indent=2)


if __name__ == "__main__":
    main()
