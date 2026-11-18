"""Check tokenizer.py and js/tokenizer.js produce identical ids. Needs Node.js.
Usage: python tests/test_tokenizer_parity.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer import encode  # noqa: E402

texts = [json.loads(line)["text"] for f in ["data/test.jsonl", "data/handwritten_test.jsonl"]
         for line in open(ROOT / f)]
texts += ["", "   ", "WHAT'S DUE", "what’s due", "café ☕ due tmrw!!", "a", "x" * 400,
          "hey " * 100, "ünïcödé only", "12:30pm on the 14th"]

js = """
const {encode} = require(process.argv[1]);
let data = ''; process.stdin.on('data', c => data += c);
process.stdin.on('end', () => { const t = JSON.parse(data); process.stdout.write(JSON.stringify(t.map(x => encode(x)))); });
"""
res = subprocess.run(["node", "-e", js, str(ROOT / "js/tokenizer.js")], input=json.dumps(texts),
                     capture_output=True, text=True, check=True)
js_ids = json.loads(res.stdout)
bad = [t for t, j in zip(texts, js_ids) if encode(t) != j]
print(f"{len(texts) - len(bad)}/{len(texts)} messages identical")
if bad:
    print("MISMATCHES:", bad[:5])
    sys.exit(1)
