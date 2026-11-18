#!/usr/bin/env bash
# Rebuild everything from scratch: data -> train -> evaluate -> profile/quantize -> tests.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore

python data/generate.py --out data --per-template 10 --seed 7
python train.py --seed 0
python evaluate.py --target 0.95
python profile_quantize.py --iters 2000
python tests/test_tokenizer_parity.py
if [ -d js/node_modules ]; then
  python tests/test_js_matches_python.py
else
  echo "skipping JS router test (run: cd js && npm install)"
fi
