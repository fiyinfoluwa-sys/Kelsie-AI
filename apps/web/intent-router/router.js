// Run Kelsie's intent router from JavaScript with ONNX Runtime (WebAssembly build).
// Works in Node (Vercel functions) and, with small changes to file loading, in the extension.
//
//   const router = await createRouter(modelPath, configPath);
//   await router.route("whats due this week")
//   -> { intent: "deadline_query", route: "local", confidence: 0.98, fellBack: false, ... }
const fs = require("fs");
const ort = require("onnxruntime-web");
const { encode, MAX_LEN } = require("./tokenizer");

function softmax(logits, temperature) {
  const scaled = Array.from(logits, (v) => v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function createRouter(modelPath, configPath) {
  ort.env.wasm.numThreads = 1;
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
  const session = await ort.InferenceSession.create(fs.readFileSync(modelPath));

  async function probs(texts) {
    const data = BigInt64Array.from(texts.flatMap((t) => encode(t)).map(BigInt));
    const ids = new ort.Tensor("int64", data, [texts.length, MAX_LEN]);
    const { logits } = await session.run({ ids });
    const n = config.labels.length;
    return texts.map((_, i) => softmax(logits.data.slice(i * n, (i + 1) * n), config.temperature));
  }

  async function route(text) {
    const [p] = await probs([text]);
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    const predicted = config.labels[best];
    const confident = p[best] >= config.threshold;
    const intent = confident ? predicted : config.fallback;
    return { text, predicted, confidence: Number(p[best].toFixed(3)), intent,
             route: config.routes[intent], fellBack: !confident };
  }

  return { route, probs, config };
}

module.exports = { createRouter };

if (require.main === module) {
  const path = require("path");
  const art = path.join(__dirname, "..", "artifacts");
  createRouter(path.join(art, "intent_int8.onnx"), path.join(art, "router_config.json")).then(async (r) => {
    const msgs = process.argv.slice(2).length ? process.argv.slice(2)
      : ["whats due this week", "save this article for my essay", "explain inflation like im 5", "delete my account"];
    for (const m of msgs) console.log(await r.route(m));
  });
}
