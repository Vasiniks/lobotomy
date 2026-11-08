#!/usr/bin/env bash
# Download a Qwen3.5 GGUF from Unsloth into models/.
#   scripts/download_model.sh                          # default: 0.8B Q4_K_M (~533 MB)
#   scripts/download_model.sh Qwen3.5-2B-Q4_K_M.gguf   # the 2B (~1.3 GB)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
FILE="${1:-Qwen3.5-0.8B-Q4_K_M.gguf}"
# The repo is named after the model size: Qwen3.5-2B-Q4_K_M.gguf -> unsloth/Qwen3.5-2B-GGUF
SIZE="$(printf '%s' "$FILE" | sed -E 's/^(Qwen3\.5-[0-9.]+B)-.*/\1/')"
URL="https://huggingface.co/unsloth/${SIZE}-GGUF/resolve/main/${FILE}"
if [ -f "models/${FILE}" ]; then
  echo "models/${FILE} already exists"
  exit 0
fi
echo "downloading ${URL}"
# Resume a partial download (-C -) and retry when the connection drops, so a
# flaky network or a re-run never starts the 533 MB over.
for attempt in 1 2 3 4 5; do
  if curl -L --fail --progress-bar -C - -o "models/${FILE}.part" "${URL}"; then
    mv "models/${FILE}.part" "models/${FILE}"
    echo "saved models/${FILE}"
    exit 0
  fi
  echo "download interrupted (attempt ${attempt} of 5); resuming in 3 s" >&2
  sleep 3
done
echo "couldn't download ${FILE}; run this script again to resume" >&2
exit 1
