#!/usr/bin/env bash
# Download required ML models for on-premise processing.
# Usage: bash download_models.sh

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== MedVoicePL — Model Download ==="
echo "Target directory: ${MODELS_DIR}"
echo ""
echo "Models will be downloaded on first run by their respective engines."
echo "  - Whisper: auto-downloaded by faster-whisper"
echo "  - SLM: auto-downloaded by transformers/huggingface_hub"
echo ""
echo "To pre-download manually, run:"
echo "  python -c \"from faster_whisper import WhisperModel; WhisperModel('large-v3')\""
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('${SLM_MODEL_ID:-speakleash/Bielik-11B-v3.0-Instruct}')\""
