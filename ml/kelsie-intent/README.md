# Kelsie intent router

A small PyTorch classifier that reads a message to Kelsie and decides where it goes
**before** any LLM call. Simple requests ("what's due this week", "start a 25 min timer",
"stop running on gmail") are handled by app code with no LLM. Requests that need
language skills go to the LLM with the right intent-specific prompt. Anything the
router is unsure about goes to the full LLM, so being unsure costs a little latency,
never a wrong action.

The int8 ONNX model is **3.0 MB** and classifies a message in **~20 µs** on one CPU
thread, plus ~36 µs for tokenizing in Python. That's tiny next to an LLM call
(hundreds of ms). It runs in Python, Node (Vercel) and the browser/extension.

## Intents

| Intent | Example | Route |
|---|---|---|
| `remember` | "save this article for my essay" | `llm_small`: short LLM call to write the note |
| `recall` | "what was that stimulus article" | `llm`: memory search + LLM answer |
| `deadline_add` | "psych paper due the 30th" | `llm_small`: extract title + date |
| `deadline_query` | "what's due this week" | `local`: database query |
| `focus` | "lock in for 45 mins" | `local`: timer |
| `application` | "cover letter for Lumen Health" | `llm` |
| `settings_privacy` | "stop running on reddit" | `local`: settings change |
| `smalltalk` | "ty kelsie" | `local`: canned reply in the user's vibe |
| `general` | "explain opportunity cost" | `llm` (also the fallback) |

Edit `intents.py` to change the intents or where they go.

> **Safety rule:** destructive settings (delete account, wipe memories) must always
> show a confirm step in the app, however confident the classifier is.

## Results

### Accuracy (honest version)

The data is **synthetic**: sentence templates with slots, typos and slang mixed in.
To avoid fooling ourselves, every intent's templates are split into train/val/test
*before* filling, so the test set only has sentence patterns the model never saw.
There's also a separate set of 72 messages written by hand (`data/handwritten_test.jsonl`).

Mean ± std over 5 full re-runs (new template split + new training seed each time,
`experiments/seed_sweep.py`):

| | Unseen templates (660) | Handwritten (72) |
|---|---|---|
| Accuracy | 76.2% ± 3.1 | 85.0% ± 4.6 |
| Macro-F1 | 76.2% ± 2.9 | 85.0% ± 4.4 |
| Handled without the fallback (coverage) | 51.6% ± 11.4 | 57.5% ± 15.4 |
| Accuracy on those | 91.8% ± 6.1 | 97.8% ± 1.4 |
| Wrong action (acted, and was wrong) | 4.1% ± 3.3 | 1.1% ± 0.6 |

The shipped model (`artifacts/`, data seed 7) scores 80.5% / 91.7% and picked a
threshold of 0.68. That's on the good side of the spread above, so plan around the averages.

**What this means:** the router is cautious, not clever. When it acts it's usually
right, and it hands about half of messages to the LLM. The ceiling is the synthetic
data. **Real messages from testers are the single biggest improvement available**
(see "Using real data").

### Confidence threshold

Softmax confidences are calibrated with temperature scaling on validation.
`evaluate.py` then picks the lowest threshold where at least 95% of the messages
the router keeps are correct on **validation**. Raise `--target` to get fewer wrong
actions and more LLM calls; lower it for the opposite.

### Profiling and quantization

One CPU thread, single message unless noted (`profile_quantize.py`, from the clean rebuild):

| Variant | Size | Test acc | Handwritten acc | Agrees with fp32 | p50 latency | Batch-64 throughput |
|---|---|---|---|---|---|---|
| PyTorch fp32 | 12.1 MB | 80.4% | 91.7% | 100% | 55 µs | 109k msg/s |
| PyTorch int8 (Linear only) | 12.0 MB | 80.8% | 90.3% | 98.5% | 89 µs | 98k msg/s |
| PyTorch int8 (Linear + embedding) | 3.3 MB | 81.2% | 91.7% | 99.1% | 108 µs | 86k msg/s |
| ONNX Runtime fp32 | 12.1 MB | 80.4% | 91.7% | 100% | 19 µs | 122k msg/s |
| **ONNX Runtime int8** | **3.0 MB** | 80.3% | 90.3% | 99.9% | **23 µs** | 118k msg/s |

Timings vary by a few µs from run to run.

What the profiler showed:

1. **The embedding table is the model.** 32,768 × 96 floats ≈ 12 MB of the 12.1 MB.
   Quantizing only the Linear layers (the usual first step) saves nothing and is
   *slower*, because quantize/dequantize overhead beats the work on such small layers.
   The size win (~4×) comes from quantizing the embedding table.
2. **Framework overhead dominates single messages.** In PyTorch eager at batch 1,
   ~44% of time is Python/dispatcher overhead around a few tiny ops. ONNX Runtime
   cuts p50 from ~55 µs to ~19 µs, so use ONNX in production.
3. **The padding-mask multiply was the top batched op.** At batch 64, `aten::mul`
   (multiplying a [64 × 128 × 96] mask) was 31% of CPU time. The padding row of the
   embedding table is always zero, so the multiply did nothing. Removing it gave
   identical outputs (max logit difference 0.0) and cut total batch-64 profile time
   232 ms → 165 ms (−29%). PyTorch fp32 throughput went from 71k to 106k msg/s and
   ONNX from 80k to 142k msg/s in that comparison run.
4. **Tokenizing costs more than the model** in Python (~36 µs vs ~20 µs). It doesn't
   matter next to an LLM call, but it's the place to look if this ever becomes a bottleneck.

**Recommendation:** ship `artifacts/intent_int8.onnx`. It's 4× smaller with the same
accuracy (within ±1 message), and it's what `js/router.js` loads.

Op tables are in `artifacts/profiling/op_tables.txt`. A Chrome trace
(`trace_fp32_batch64.json`, open in `chrome://tracing` or Perfetto) is written there too.

### JavaScript parity

- `js/tokenizer.js` produces **identical** token ids to `tokenizer.py` on 742 messages,
  including edge cases (curly apostrophes, emoji, accents, very long input).
- `js/router.js` (onnxruntime-web, int8): the fp32 model gives identical probabilities to
  Python. With int8, WebAssembly and native kernels round slightly differently (max
  probability difference 0.028). That flips 5 coin-flip predictions, all below the
  threshold, and 2 messages sitting right at the threshold (one side keeps, the other
  falls back). **Conflicting actions: 0.**

## How it works

- `tokenizer.py`: fastText-style hashed features (words, word pairs, letter trigrams)
  → FNV-1a hash → 32,768 buckets. No vocab file, and letter trigrams make it tolerant of typos.
- `model.py`: embedding → mean over tokens → LayerNorm → 96→128→9 MLP. ~3.2M parameters,
  almost all in the embedding table.
- `train.py`: AdamW, label smoothing, 10% feature dropout, early stopping on validation
  macro-F1, then temperature scaling. ~20 s on one CPU core.

## Run it

```bash
pip install -r requirements.txt
cd js && npm install && cd ..        # only for the JS router + its test
./run_all.sh                          # data -> train -> evaluate -> profile/quantize -> tests
python router.py "whats due tmrw"     # try the Python router
node js/router.js "save this for my essay"   # try the JS router
python experiments/seed_sweep.py      # stability check (~3 min)
```

## Using it in Kelsie

- **Vercel (Node):** copy `js/tokenizer.js`, `js/router.js`, `artifacts/intent_int8.onnx`
  and `artifacts/router_config.json` into the backend. Create the router once per
  function instance (not per request), then `await router.route(text)` and switch on
  `result.route`. `onnxruntime-node` also works if your build can download its binary.
- **Chrome extension:** same files with `onnxruntime-web`. Load the model with
  `fetch(chrome.runtime.getURL(...))` instead of `fs`. The WebAssembly files must be
  packaged inside the extension (Manifest V3 doesn't allow remote code).
- **Keep the tokenizers in sync:** after any change to `tokenizer.py` or
  `js/tokenizer.js`, run `python tests/test_tokenizer_parity.py`.

## Using real data

1. With testers' consent, log the message, the router's choice and what actually
   happened (did the user undo it? rephrase? did the LLM do something else?).
   Treat these logs as sensitive: store them with the same RLS rules, strip names,
   emails and links, and let people opt out.
2. Label a few hundred real messages per intent (start with the ones the router
   sent to the fallback; they're the most informative).
3. Add them to `train.jsonl` (keep a real-data test set aside that's never trained on),
   re-run `./run_all.sh`, and compare against the numbers above.

## Limitations and next steps

- Synthetic data → ~76–85% accuracy. Real data is the fix.
- No conversation context: "yes do it" or "the second one" can't be routed alone.
  Pass the previous intent along, or send short follow-ups to the LLM.
- English only.
- If you need more accuracy and can afford ~20 MB and a few ms, fine-tune a small
  pretrained sentence encoder (e.g. a MiniLM-sized model) on the same data. It
  generalises much better from few examples. That wasn't possible here because
  model downloads were blocked in the build environment.
