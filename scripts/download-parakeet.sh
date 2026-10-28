#!/usr/bin/env sh
# Download the optional Parakeet v3 STT model (~675 MB).
# Usage: scripts/download-parakeet.sh [models-dir]
# Then run voicer with: --stt-model parakeet
set -e
DIR="${1:-models}"
mkdir -p "$DIR/parakeet"
B="https://huggingface.co/ysdede/parakeet-tdt-0.6b-v3-onnx/resolve/main"

echo ">> Parakeet TDT 0.6B v3 (STT, multilingual, ONNX int8) ~675 MB"
curl -L --fail -o "$DIR/parakeet/nemo128.onnx" "$B/nemo128.onnx"
curl -L --fail -o "$DIR/parakeet/decoder_joint-model.int8.onnx" "$B/decoder_joint-model.int8.onnx"
curl -L --fail -o "$DIR/parakeet/vocab.txt" "$B/vocab.txt"
curl -L --fail -o "$DIR/parakeet/encoder-model.int8.onnx" "$B/encoder-model.int8.onnx"

echo "Done. Run with: voicer --stt-model parakeet"
