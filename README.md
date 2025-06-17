# qwen3-embedding-tei-onnx

Convert Qwen3-Embedding-0.6B to ONNX format for [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference).

## Quick Start

```bash
./run_onnx_conversion.sh
```

This script will prompt you to choose between:
- **Standard ONNX (float32)**: Maximum accuracy, ~4.7GB
- **Quantized ONNX (int8)**: 2-4x faster CPU inference, ~1.2GB

## Requirements

- Python 3.8+
- 8GB+ RAM
- [uv](https://github.com/astral-sh/uv) (optional, for faster installs)

## What it does

- Converts Qwen3-Embedding-0.6B to ONNX with mean pooling
- Consolidates weights into single `model.onnx_data` file (TEI requirement)
- Fixes dynamic axes for proper batching
- Validates the converted model
- Optionally quantizes to INT8 for faster CPU inference

## Technical Notes

- Exports raw hidden states (3D tensor) without pooling - TEI applies pooling based on config
- Uses temporary directory during export to ensure clean output
- Consolidates all weights into single `model.onnx_data` file as required by TEI

## Using with TEI

### Standard Model
```bash
# Upload to HuggingFace
huggingface-cli upload YOUR_USERNAME/qwen3-embedding-tei-onnx ./qwen3-tei-onnx

# Run with TEI
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id YOUR_USERNAME/qwen3-embedding-tei-onnx
```

### INT8 Quantized Model (CPU optimized)
```bash
# Upload to HuggingFace
huggingface-cli upload YOUR_USERNAME/qwen3-embedding-int8-tei-onnx ./qwen3-tei-onnx-int8

# Run with TEI on CPU
docker run -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id YOUR_USERNAME/qwen3-embedding-int8-tei-onnx
```

## License

Apache 2.0