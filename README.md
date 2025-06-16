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

## Important Note on Model Size

The ONNX export uses `do_constant_folding=False` to prevent weight duplication. The Qwen3 model has `tie_word_embeddings=true` in its configuration, which means the embedding weights are shared between input and output layers in the original PyTorch model. When `do_constant_folding=True` is used during ONNX export, it can break this weight sharing and duplicate the weights, effectively doubling the model size from ~4.7GB to ~8.9GB.

By setting `do_constant_folding=False`, we preserve the model's original size and prevent unnecessary weight duplication.

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