#!/bin/bash

echo "🚀 Qwen3-Embedding-0.6B to TEI-compatible ONNX Converter"
echo "========================================================"

# Check if virtual environment exists
if [ ! -d "venv_onnx" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_onnx
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv_onnx/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements_onnx_export.txt

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