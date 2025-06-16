# Qwen3-Embedding-0.6B ONNX Conversion for TEI

This pipeline converts Qwen3-Embedding-0.6B to ONNX format compatible with Text Embeddings Inference (TEI).

## Files

- `convert_qwen3_tei_onnx.py` - Main conversion script that handles TEI-specific requirements
- `requirements_onnx_export.txt` - Python dependencies
- `run_onnx_conversion.sh` - Easy-to-use wrapper script

## Usage

Simply run:
```bash
./run_onnx_conversion.sh
```

This will:
1. Create a virtual environment
2. Install dependencies
3. Convert the model to ONNX with TEI-compatible structure
4. Output to `./qwen3-tei-onnx/`

## Key Features

- Converts Qwen3-Embedding-0.6B to ONNX format
- Consolidates external data into single `model.onnx_data` file (TEI requirement)
- Includes mean pooling in the ONNX model
- Creates `config_sentence_transformers.json` for TEI compatibility
- Verifies the converted model works with ONNX Runtime

## Output Structure

```
qwen3-tei-onnx/
├── model.onnx              # ONNX graph (small file ~1.5MB)
├── model.onnx_data         # Consolidated weights (~2.3GB)
├── config.json             # Model configuration
├── config_sentence_transformers.json  # TEI-specific config
├── tokenizer.json          # Tokenizer
└── ... (other tokenizer files)
```

## Using with TEI

After conversion, either:

1. Upload to HuggingFace Hub:
   ```bash
   huggingface-cli upload YOUR_USERNAME/qwen3-embedding-0.6b-tei-onnx ./qwen3-tei-onnx
   ```

2. Update porter.yaml:
   ```yaml
   model-id: YOUR_USERNAME/qwen3-embedding-0.6b-tei-onnx
   ```

## Notes

- The conversion properly handles the 2GB ONNX limit by using external data storage
- All weights are consolidated into a single `model.onnx_data` file as TEI expects
- The model includes mean pooling built into the ONNX graph