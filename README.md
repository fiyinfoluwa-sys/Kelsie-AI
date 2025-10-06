# Kelsie-AI — README

Kelsie is a small, local chatbot you can run from the command line. It’s built with Hugging Face Transformers and tuned to be easy to run on a Mac (MPS) or CPU. This README is short, human, and focused on getting you chatting quickly.

---

## Quick summary
- Run Kelsie locally (no paid APIs required).
- Works on macOS (MPS) or CPU.
- Swap models quickly via an environment variable `KELSIE_MODEL`.
- Lightweight defaults so it won’t normally crash your machine.

---

## Files you care about
- `backend/kelsie_cli_transformers.py` — the main CLI script (one file you run).
- `requirements.txt` — Python dependencies.
- `README.md` — (this file).

---

## Setup (5 minutes)

1. Clone your repo (use your SSH remote):
```bash
git clone git@github.com:fiyinfoluwa-sys/Kelsie-AI.git
cd Kelsie-AI

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt
# Optional for quantization/large models:
pip install bitsandbytes accelerate

4. RUN KELSIE
export KELSIE_MODEL="microsoft/DialoGPT-medium"
python backend/kelsie_cli_transformers.py

Type a message after 'You:' and press Enter. Type 'quit' to exit. 
