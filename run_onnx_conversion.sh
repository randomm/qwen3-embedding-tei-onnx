#!/bin/bash

echo "🚀 Qwen3-Embedding-0.6B to TEI-compatible ONNX Converter"
echo "========================================================"

# Remove old virtual environment if it exists
if [ -d ".venv" ]; then
    echo "🧹 Removing old virtual environment..."
    rm -rf .venv
fi

# Create fresh virtual environment with uv
echo "📦 Creating virtual environment with uv..."
uv venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv
echo "📥 Installing dependencies with uv..."
uv pip install -r requirements_onnx_export.txt

# Run the conversion
echo ""
echo "🔄 Starting ONNX conversion..."
echo "This may take a few minutes..."
python convert_qwen3_tei_onnx.py

# Check if conversion was successful
if [ -f "./qwen3-tei-onnx/model.onnx" ]; then
    echo ""
    echo "✅ Conversion successful!"
    echo ""
    echo "📊 Model info:"
    ls -lh ./qwen3-tei-onnx/model.onnx
    echo ""
    echo "🚀 To use with TEI:"
    echo "1. Upload to HuggingFace Hub:"
    echo "   huggingface-cli login"
    echo "   huggingface-cli upload YOUR_USERNAME/qwen3-embedding-0.6b-tei-onnx ./qwen3-tei-onnx"
    echo ""
    echo "2. Update porter.yaml:"
    echo "   model-id: YOUR_USERNAME/qwen3-embedding-0.6b-tei-onnx"
else
    echo ""
    echo "❌ Conversion failed. Check the error messages above."
fi

# Deactivate virtual environment
deactivate