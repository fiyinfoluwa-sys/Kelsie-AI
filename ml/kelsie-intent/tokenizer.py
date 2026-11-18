"""Hashed n-gram tokenizer (fastText-style). No vocab file needed.

Features per message:
  w:<word>          every word
  b:<w1>_<w2>       every adjacent word pair
  c:<tri>           character trigrams of "<word>" (handles typos like "rmeind")
Each feature is hashed with 32-bit FNV-1a into one of NUM_BUCKETS-1 ids
(id 0 is reserved for padding).

Kept deliberately simple so it can be ported 1:1 to JavaScript (see tokenizer.js);
tests/test_tokenizer_parity.py checks the two produce identical ids.
"""
import re

NUM_BUCKETS = 32768
MAX_LEN = 128

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


def fnv1a(s: str) -> int:
    h = _FNV_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def normalize(text: str) -> list[str]:
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


def features(text: str) -> list[str]:
    words = normalize(text)
    feats = ["w:" + w for w in words]
    feats += ["b:" + a + "_" + b for a, b in zip(words, words[1:])]
    for w in words:
        padded = "<" + w + ">"
        feats += ["c:" + padded[i:i + 3] for i in range(len(padded) - 2)]
    return feats


def encode(text: str, max_len: int = MAX_LEN) -> list[int]:
    ids = [(fnv1a(f) % (NUM_BUCKETS - 1)) + 1 for f in features(text)][:max_len]
    if not ids:
        ids = [0]
    return ids + [0] * (max_len - len(ids))
