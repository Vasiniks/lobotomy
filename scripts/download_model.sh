#!/usr/bin/env bash
# Download the default model: Unsloth's Qwen3.5 0.8B, Q4_K_M GGUF (~533 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
FILE="${1:-Qwen3.5-0.8B-Q4_K_M.gguf}"
URL="https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/${FILE}"
if [ -f "models/${FILE}" ]; then
  echo "models/${FILE} already exists"
  exit 0
fi
echo "downloading ${URL}"
curl -L --fail --progress-bar -o "models/${FILE}.part" "${URL}"
mv "models/${FILE}.part" "models/${FILE}"
echo "saved models/${FILE}"
