# qwen3-embedding-tei-onnx

Convert Qwen3-Embedding-0.6B to ONNX format for [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference).

## Quick Start

```bash
./run_onnx_conversion.sh
```

This creates a TEI-compatible ONNX model in `./qwen3-tei-onnx/`.

## Requirements

- Python 3.8+
- 8GB+ RAM
- [uv](https://github.com/astral-sh/uv) (optional, for faster installs)

## What it does

- Converts Qwen3-Embedding-0.6B to ONNX with mean pooling
- Consolidates weights into single `model.onnx_data` file (TEI requirement)
- Fixes dynamic axes for proper batching
- Validates the converted model

## Technical Notes

- Exports raw hidden states (3D tensor) without pooling - TEI applies pooling based on config
- Uses temporary directory during export to ensure clean output
- Consolidates all weights into single `model.onnx_data` file as required by TEI

## Using with TEI

```bash
# Upload to HuggingFace
huggingface-cli upload YOUR_USERNAME/qwen3-embedding-tei-onnx ./qwen3-tei-onnx

# Run with TEI
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id YOUR_USERNAME/qwen3-embedding-tei-onnx
```

## License

Apache 2.0