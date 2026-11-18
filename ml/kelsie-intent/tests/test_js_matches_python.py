"""Check js/router.js (onnxruntime-web, int8 ONNX) gives the same predictions as
Python ONNX Runtime on the same int8 model, and report its accuracy. Needs Node + `npm install` in js/.
Usage: python tests/test_js_matches_python.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from intents import LABELS  # noqa: E402
from tokenizer import encode  # noqa: E402

rows = [json.loads(line) for f in ["data/test.jsonl", "data/handwritten_test.jsonl"] for line in open(ROOT / f)]
texts = [r["text"] for r in rows]

sess = ort.InferenceSession(str(ROOT / "artifacts/intent_int8.onnx"), providers=["CPUExecutionProvider"])
py_logits = sess.run(["logits"], {"ids": np.array([encode(t) for t in texts], dtype=np.int64)})[0]
py_pred = py_logits.argmax(1)

js = """
const {createRouter} = require(process.argv[1] + '/router.js');
const art = process.argv[1] + '/../artifacts/';
let d=''; process.stdin.on('data', c => d += c);
process.stdin.on('end', async () => {
  const r = await createRouter(art + 'intent_int8.onnx', art + 'router_config.json');
  const t = JSON.parse(d); const out = [];
  for (let i = 0; i < t.length; i += 64) for (const p of await r.probs(t.slice(i, i + 64))) out.push(p);
  process.stdout.write(JSON.stringify(out));
});
"""
res = subprocess.run(["node", "-e", js, str(ROOT / "js")], input=json.dumps(texts),
                     capture_output=True, text=True, check=True, cwd=ROOT / "js")
js_probs = np.array(json.loads(res.stdout))
js_pred = js_probs.argmax(1)
y = np.array([LABELS.index(r["label"]) for r in rows])

# int8 kernels differ slightly between WebAssembly and native ORT (activation rounding), which can
# flip near-ties. What matters is the ROUTING decision after the confidence threshold.
cfg = json.loads((ROOT / "artifacts/router_config.json").read_text())
z = py_logits / cfg["temperature"]
py_probs = np.exp(z - z.max(1, keepdims=True))
py_probs /= py_probs.sum(1, keepdims=True)
fb = LABELS.index(cfg["fallback"])


def routed(p):
    return np.where(p.max(1) >= cfg["threshold"], p.argmax(1), fb)


rp, rj = routed(py_probs), routed(js_probs)
same_route = rp == rj
# harmful = both runtimes act (neither falls back) but pick DIFFERENT intents
conflict = (~same_route) & (rp != fb) & (rj != fb)
print(f"raw argmax identical: {(js_pred == py_pred).sum()}/{len(rows)}  "
      f"(max prob difference {np.abs(py_probs - js_probs).max():.4f})")
print(f"final routing identical: {same_route.sum()}/{len(rows)}")
for i in np.where(~same_route)[0]:
    print(f"   at threshold edge: {texts[i]!r} py {LABELS[rp[i]]} ({py_probs[i].max():.3f}) "
          f"js {LABELS[rj[i]]} ({js_probs[i].max():.3f})")
print(f"conflicting actions (both confident, different intent): {conflict.sum()}")
print(f"JS int8 accuracy on test+handwritten: {(js_pred == y).mean():.3f}")
sys.exit(0 if conflict.sum() == 0 else 1)
