#!/usr/bin/env sh
# Download all models needed by voicer (~100 MB total).
# Usage: scripts/download-models.sh [models-dir]
set -e
DIR="${1:-models}"
mkdir -p "$DIR/moonshine" "$DIR/vad" "$DIR/tts"

echo ">> Moonshine tiny (STT, English, ONNX) ~32 MB"
B="https://huggingface.co/onnx-community/moonshine-tiny-ONNX/resolve/main"
curl -L --fail -o "$DIR/moonshine/encoder_model_quantized.onnx" "$B/onnx/encoder_model_quantized.onnx"
curl -L --fail -o "$DIR/moonshine/decoder_model_quantized.onnx" "$B/onnx/decoder_model_quantized.onnx"
curl -L --fail -o "$DIR/moonshine/decoder_with_past_model_quantized.onnx" "$B/onnx/decoder_with_past_model_quantized.onnx"
curl -L --fail -o "$DIR/moonshine/tokenizer.json" "$B/tokenizer.json"
curl -L --fail -o "$DIR/moonshine/config.json" "$B/config.json"
curl -L --fail -o "$DIR/moonshine/generation_config.json" "$B/generation_config.json"

echo ">> Silero VAD ~2.3 MB"
curl -L --fail -o "$DIR/vad/silero_vad.onnx" \
  "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

echo ">> Piper voice: en_US amy medium ~63 MB"
V="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
curl -L --fail -o "$DIR/tts/en_US-amy-medium.onnx" "$V/en_US-amy-medium.onnx"
curl -L --fail -o "$DIR/tts/en_US-amy-medium.onnx.json" "$V/en_US-amy-medium.onnx.json"

echo "Done. Models in: $DIR"
