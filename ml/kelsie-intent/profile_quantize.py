"""Profile the classifier and compare quantized variants.

Variants
  fp32            original PyTorch model
  int8_linear     PyTorch dynamic int8 on the Linear layers only
  int8_full       + int8 weight-only embedding table (most of the model's size)
  onnx_fp32       ONNX export, run with ONNX Runtime
  onnx_int8       ONNX Runtime dynamic int8 (MatMul + Gather/embedding)

For each: file size, accuracy on test + handwritten, agreement with fp32,
single-message latency (p50/p95, model only and with tokenization), and batch-64
throughput. All timings use 1 CPU thread to mimic a small serverless instance.
Also writes torch.profiler op tables and a Chrome trace.

Usage: python profile_quantize.py [--iters 2000]
"""
import argparse
import copy
import io
import json
import os
import statistics
import time
import warnings

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic as ort_quantize_dynamic
from torch.ao.quantization import (convert, float_qparams_weight_only_qconfig, prepare,
                                   quantize_dynamic)
from torch.profiler import ProfilerActivity, profile, record_function

from common import ARTIFACTS, ROOT, load_jsonl, load_model, to_tensors
from tokenizer import MAX_LEN, encode

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")


def state_dict_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def make_int8_linear(fp32):
    return quantize_dynamic(copy.deepcopy(fp32), {torch.nn.Linear}, dtype=torch.qint8)


def make_int8_full(fp32):
    m = copy.deepcopy(fp32)
    m.embed.padding_idx = None  # quantized Embedding has no padding_idx; row 0 is zero and stays ~zero
    m.embed.qconfig = float_qparams_weight_only_qconfig
    prepare(m, inplace=True)
    convert(m, inplace=True)
    return quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)


def export_onnx(fp32, path):
    dummy = torch.tensor([encode("whats due this week")], dtype=torch.long)
    torch.onnx.export(fp32, (dummy,), str(path), input_names=["ids"], output_names=["logits"],
                      dynamic_axes={"ids": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)


class TorchRunner:
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, ids_np):
        return self.model(torch.from_numpy(ids_np)).numpy()


class OrtRunner:
    def __init__(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, ids_np):
        return self.sess.run(["logits"], {"ids": ids_np})[0]


def latency(runner, texts, iters, with_tokenizer):
    pre = [np.array([encode(t)], dtype=np.int64) for t in texts]
    for i in range(50):  # warm-up
        runner(pre[i % len(pre)])
    times = []
    for i in range(iters):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        ids = np.array([encode(t)], dtype=np.int64) if with_tokenizer else pre[i % len(pre)]
        runner(ids)
        times.append((time.perf_counter() - start) * 1e6)
    times.sort()
    return {"p50_us": round(statistics.median(times), 1), "p95_us": round(times[int(0.95 * len(times))], 1),
            "mean_us": round(statistics.fmean(times), 1)}


def throughput(runner, x_np, reps=30):
    batch = x_np[:64]
    for _ in range(5):
        runner(batch)
    start = time.perf_counter()
    for _ in range(reps):
        runner(batch)
    return round(reps * len(batch) / (time.perf_counter() - start))


def accuracy(runner, x_np, y_np):
    return float((runner(x_np).argmax(1) == y_np).mean())


def torch_profile(name, model, x):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(200):
                with record_function(f"{name}_forward"):
                    model(x)
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
    return prof, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()
    out_dir = ARTIFACTS / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32, ckpt = load_model()
    test_rows = load_jsonl(ROOT / "data/test.jsonl")
    hw_rows = load_jsonl(ROOT / "data/handwritten_test.jsonl")
    xt, yt = (a.numpy() for a in to_tensors(test_rows))
    xh, yh = (a.numpy() for a in to_tensors(hw_rows))
    texts = [r["text"] for r in test_rows]

    # --- build variants ---
    int8_linear = make_int8_linear(fp32)
    int8_full = make_int8_full(fp32)
    onnx_fp32 = ARTIFACTS / "intent_fp32.onnx"
    onnx_int8 = ARTIFACTS / "intent_int8.onnx"
    export_onnx(fp32, onnx_fp32)
    ort_quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "Gemm", "Gather"])
    torch.save(int8_full.state_dict(), ARTIFACTS / "intent_int8_full.pt")

    variants = {
        "fp32": (TorchRunner(fp32), state_dict_bytes(fp32)),
        "int8_linear": (TorchRunner(int8_linear), state_dict_bytes(int8_linear)),
        "int8_full": (TorchRunner(int8_full), state_dict_bytes(int8_full)),
        "onnx_fp32": (OrtRunner(onnx_fp32), onnx_fp32.stat().st_size),
        "onnx_int8": (OrtRunner(onnx_int8), onnx_int8.stat().st_size),
    }

    # --- tokenizer cost on its own ---
    t0 = time.perf_counter()
    for i in range(args.iters):
        encode(texts[i % len(texts)])
    tok_us = (time.perf_counter() - t0) / args.iters * 1e6

    ref_test = variants["fp32"][0](xt).argmax(1)
    ref_hw = variants["fp32"][0](xh).argmax(1)
    results = {"tokenizer_mean_us": round(tok_us, 1), "threads": 1, "max_len": MAX_LEN, "variants": {}}
    for name, (runner, size) in variants.items():
        r = {
            "size_kb": round(size / 1024, 1),
            "test_acc": round(accuracy(runner, xt, yt), 4),
            "handwritten_acc": round(accuracy(runner, xh, yh), 4),
            "agree_with_fp32_test": round(float((runner(xt).argmax(1) == ref_test).mean()), 4),
            "agree_with_fp32_handwritten": round(float((runner(xh).argmax(1) == ref_hw).mean()), 4),
            "latency_model_only": latency(runner, texts, args.iters, with_tokenizer=False),
            "latency_with_tokenizer": latency(runner, texts, args.iters, with_tokenizer=True),
            "throughput_msgs_per_s_batch64": throughput(runner, xt),
        }
        results["variants"][name] = r
        print(f"{name:12s} {r['size_kb']:8.1f} KB  test {r['test_acc']:.3f}  hw {r['handwritten_acc']:.3f}  "
              f"agree {r['agree_with_fp32_test']:.3f}  p50 {r['latency_model_only']['p50_us']:7.1f}us  "
              f"p95 {r['latency_model_only']['p95_us']:7.1f}us  batch64 {r['throughput_msgs_per_s_batch64']}/s")
    print(f"tokenizer: {tok_us:.1f} us/message")

    # --- torch.profiler: where does time go? ---
    one = torch.from_numpy(xt[:1])
    batch = torch.from_numpy(xt[:64])
    tables = {}
    for name, model in [("fp32", fp32), ("int8_full", int8_full)]:
        for label, x in [("batch1", one), ("batch64", batch)]:
            prof, table = torch_profile(f"{name}_{label}", model, x)
            tables[f"{name}_{label}"] = table
            if name == "fp32" and label == "batch64":
                prof.export_chrome_trace(str(out_dir / "trace_fp32_batch64.json"))
    with open(out_dir / "op_tables.txt", "w") as f:
        for k, v in tables.items():
            f.write(f"===== {k} =====\n{v}\n\n")
    print("\n" + tables["fp32_batch1"])
    print(tables["fp32_batch64"])

    (ARTIFACTS / "quantization_report.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
