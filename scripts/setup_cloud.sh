#!/bin/bash
# AATQ Cloud GPU Setup Script
# Usage: bash scripts/setup_cloud.sh [--model llama2-7b|qwen2-7b|...]
set -e

# Parse arguments
MODEL="qwen2-7b"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        *) MODEL="$1"; shift;;
    esac
done

echo "=================================="
echo "  AATQ Cloud GPU Environment Setup"
echo "=================================="

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
pip install transformers safetensors datasets numpy accelerate

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

echo ""
echo "Pre-downloading model (avoid download time during experiment)..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

# Map model keys to HF paths (same as MODELS registry in run_7b_all.py)
MODELS = {
    'qwen2-7b':   'Qwen/Qwen2.5-7B',
    'llama2-7b':  'meta-llama/Llama-2-7b-hf',
    'llama3-8b':  'meta-llama/Meta-Llama-3-8B',
    'tinyllama':  'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
}

model_key = '$MODEL'

hf_path = MODELS.get(model_key)
if not hf_path:
    # Assume it's a raw HF path
    hf_path = model_key

print(f'Model: {hf_path}')

# Check HF token for gated models
if 'llama' in hf_path.lower() or 'meta-' in hf_path.lower():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        from huggingface_hub import login
        login(token)
        print('HF token set for gated model access')
    else:
        print('WARNING: HF_TOKEN not set. LLaMA models require HuggingFace authentication.')
        print('  1. Create account at huggingface.co')
        print('  2. Accept Meta license at huggingface.co/meta-llama/Llama-2-7b-hf')
        print('  3. Create token at huggingface.co/settings/tokens')
        print('  4. Run: export HF_TOKEN=hf_your_token_here')

tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
print(f'Tokenizer OK (vocab={tokenizer.vocab_size})')
m = AutoModelForCausalLM.from_pretrained(
    hf_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
print(f'Model OK, GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB')
del m; torch.cuda.empty_cache()
print('Pre-download complete, cache ready.')
"

echo ""
echo "=================================="
echo "  Setup complete!"
echo "  Run: python scripts/run_7b_all.py --model $MODEL --analyze-layers --skip-ste"
echo "=================================="
