import json
from pathlib import Path

import torch

from intents import LABEL2ID, LABELS
from model import IntentClassifier
from tokenizer import encode

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_tensors(rows):
    x = torch.tensor([encode(r["text"]) for r in rows], dtype=torch.long)
    y = torch.tensor([LABEL2ID[r["label"]] for r in rows], dtype=torch.long)
    return x, y


def load_model(path=ARTIFACTS / "intent_fp32.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = IntentClassifier(num_labels=len(LABELS), **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
