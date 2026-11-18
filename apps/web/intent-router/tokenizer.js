// JavaScript port of tokenizer.py. Must stay byte-for-byte identical in output:
// run `python tests/test_tokenizer_parity.py` after changing either file.
const NUM_BUCKETS = 32768;
const MAX_LEN = 128;

function fnv1a(str) {
  let h = 0x811c9dc5;
  const bytes = new TextEncoder().encode(str);
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function normalize(text) {
  const t = text.toLowerCase().replace(/'/g, "").replace(/’/g, "").replace(/[^a-z0-9]+/g, " ");
  return t.split(" ").filter(Boolean);
}

function features(text) {
  const words = normalize(text);
  const feats = words.map((w) => "w:" + w);
  for (let i = 0; i < words.length - 1; i++) feats.push("b:" + words[i] + "_" + words[i + 1]);
  for (const w of words) {
    const p = "<" + w + ">";
    for (let i = 0; i < p.length - 2; i++) feats.push("c:" + p.slice(i, i + 3));
  }
  return feats;
}

function encode(text, maxLen = MAX_LEN) {
  let ids = features(text).map((f) => (fnv1a(f) % (NUM_BUCKETS - 1)) + 1).slice(0, maxLen);
  if (ids.length === 0) ids = [0];
  while (ids.length < maxLen) ids.push(0);
  return ids;
}

module.exports = { encode, features, normalize, fnv1a, NUM_BUCKETS, MAX_LEN };
