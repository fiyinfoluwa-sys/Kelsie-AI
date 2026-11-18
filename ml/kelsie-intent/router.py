"""Route a message: classify, then decide where it goes.

If the classifier's calibrated confidence is below the threshold, the message
goes to the full LLM ("general") instead of a local handler. The LLM can handle
anything, so being unsure costs a bit of latency/money, never a wrong action.

Destructive settings (delete account, wipe memories) must ALWAYS show a confirm
step in the app, no matter how confident the classifier is.
"""
import json

import torch

from common import ARTIFACTS, load_model
from intents import FALLBACK, LABELS, ROUTE
from tokenizer import encode


class Router:
    def __init__(self, model=None, temperature=None, threshold=None):
        if model is None:
            model, ckpt = load_model()
            temperature = ckpt["temperature"]
        self.model = model
        self.temperature = temperature or 1.0
        if threshold is None:
            threshold = json.loads((ARTIFACTS / "threshold.json").read_text())["threshold"]
        self.threshold = threshold

    @torch.no_grad()
    def probs(self, texts):
        x = torch.tensor([encode(t) for t in texts], dtype=torch.long)
        return torch.softmax(self.model(x) / self.temperature, dim=-1)

    def route(self, text):
        p = self.probs([text])[0]
        conf, idx = float(p.max()), int(p.argmax())
        intent = LABELS[idx]
        confident = conf >= self.threshold
        final = intent if confident else FALLBACK
        return {"text": text, "predicted": intent, "confidence": round(conf, 3),
                "intent": final, "route": ROUTE[final], "fell_back": not confident}


if __name__ == "__main__":
    import sys
    r = Router()
    for t in sys.argv[1:] or ["whats due this week", "save this article", "explain inflation"]:
        print(r.route(t))
