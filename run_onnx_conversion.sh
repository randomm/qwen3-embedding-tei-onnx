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

# Install pip-tools for requirements management
echo "📥 Installing pip-tools..."
uv pip install pip-tools

# Compile requirements.in to requirements.txt
echo "🔄 Compiling requirements..."
uv pip compile requirements.in -o requirements.txt

# Install dependencies with uv
echo "📥 Installing dependencies with uv..."
uv pip install -r requirements.txt

# Run the unified conversion script
echo ""
echo "🔄 Starting ONNX conversion..."
echo "This script will prompt you to choose between standard and quantized models."
python convert_unified.py

# Check if conversion was successful (check both possible output directories)
if [ -f "./qwen3-tei-onnx/model.onnx" ] || [ -f "./qwen3-tei-onnx-int8/model.onnx" ]; then
    echo ""
    echo "✅ Conversion successful!"
    echo ""
    echo "📊 Model info:"
    if [ -f "./qwen3-tei-onnx/model.onnx" ]; then
        echo "Standard model:"
        ls -lh ./qwen3-tei-onnx/model.onnx*
    fi
    if [ -f "./qwen3-tei-onnx-int8/model.onnx" ]; then
        echo "INT8 quantized model:"
        ls -lh ./qwen3-tei-onnx-int8/model.onnx*
    fi
    echo ""
    echo "🚀 To use with TEI:"
    echo "1. Upload to HuggingFace Hub:"
    echo "   huggingface-cli login"
    if [ -f "./qwen3-tei-onnx/model.onnx" ]; then
        echo "   huggingface-cli upload YOUR_USERNAME/qwen3-embedding-0.6b-tei-onnx ./qwen3-tei-onnx"
    fi
    if [ -f "./qwen3-tei-onnx-int8/model.onnx" ]; then
        echo "   huggingface-cli upload YOUR_USERNAME/qwen3-embedding-0.6b-int8-tei-onnx ./qwen3-tei-onnx-int8"
    fi
    echo ""
    echo "2. Update porter.yaml with your model ID"
else
    echo ""
    echo "❌ Conversion failed. Check the error messages above."
fi

# Deactivate virtual environment
deactivate