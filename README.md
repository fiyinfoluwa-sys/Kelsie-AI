# Kelsie AI 🤖

A powerful, web-connected AI chatbot built with Python and Hugging Face transformers. Kelsie combines local AI processing with real-time Google Search integration for accurate, up-to-date information.

## Features ✨

- **Web-Connected Intelligence**: Real-time Google Search integration for current information
- **Local AI Processing**: Uses Microsoft DialoGPT-medium model running locally
- **Conversation Memory**: Maintains context across conversation turns
- **Apple Silicon Optimized**: Automatic MPS detection for faster inference
- **Free & Open Source**: No API costs for the core functionality
- **Easy Setup**: Simple installation and configuration

## Tech Stack 🛠️

- **Python 3.9+**
- **PyTorch** - AI model inference
- **Hugging Face Transformers** - Pre-trained language models
- **Google Custom Search API** - Real-time web search
- **Requests** - HTTP library for API calls

## Installation 📦

1. **Clone the repository**
```bash
git clone https://github.com/fiyinfoluwa-sys/Kelsie-AI.git
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
