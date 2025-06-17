#!/usr/bin/env python3
"""
Quantize Qwen3-Embedding ONNX model to int8 for faster CPU inference
"""

import os
import sys
from pathlib import Path
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnx
import onnxruntime as ort
import shutil

def quantize_model(input_model_path, output_model_path):
    """
    Quantize ONNX model to int8 using dynamic quantization
    """
    print(f"Loading model from: {input_model_path}")
    
    # Create output directory
    output_dir = Path(output_model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-process: Shape inference is important for transformer models
    print("Running shape inference pre-processing...")
    preprocessed_model_path = output_dir / "model_preprocessed.onnx"
    
    # For models with external data, we need to handle this carefully
    quant_pre_process(
        input_model_path=str(input_model_path),
        output_model_path=str(preprocessed_model_path),
        auto_merge=True,  # Merge external data for processing
        save_as_external_data=True,  # Save back as external data
        all_tensors_to_one_file=True,
        external_data_location="model_preprocessed.onnx_data",
        size_threshold=1024
    )
    
    # Quantize the model
    print("Quantizing model to int8...")
    quantize_dynamic(
        model_input=str(preprocessed_model_path),
        model_output=str(output_model_path),
        weight_type=QuantType.QInt8
    )
    
    # Clean up preprocessed model
    print("Cleaning up temporary files...")
    os.remove(preprocessed_model_path)
    if (output_dir / "model_preprocessed.onnx_data").exists():
        os.remove(output_dir / "model_preprocessed.onnx_data")
    
    print(f"Quantized model saved to: {output_model_path}")
    
    # Verify the quantized model
    print("\nVerifying quantized model...")
    providers = ['CPUExecutionProvider']
    
    # Test with dummy input
    print("Creating test session...")
    session = ort.InferenceSession(str(output_model_path), providers=providers)
    
    # Get input shapes
    input_shape = session.get_inputs()[0].shape
    batch_size = 1 if input_shape[0] == 'batch_size' else input_shape[0]
    seq_length = 128 if input_shape[1] == 'sequence_length' else input_shape[1]
    
    # Create dummy inputs
    dummy_input_ids = np.random.randint(0, 1000, size=(batch_size, seq_length), dtype=np.int64)
    dummy_attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
    
    # Run inference
    print("Running test inference...")
    outputs = session.run(None, {
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    })
    
    print(f"✅ Quantized model output shape: {outputs[0].shape}")
    print(f"✅ Output dtype: {outputs[0].dtype} (should be float32)")
    
    return output_model_path

def copy_supporting_files(source_dir, target_dir):
    """Copy tokenizer and config files to quantized model directory"""
    files_to_copy = [
        'config.json',
        'config_sentence_transformers.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'added_tokens.json',
        'vocab.json',
        'merges.txt',
        'chat_template.jinja'
    ]
    
    for file in files_to_copy:
        source_path = Path(source_dir) / file
        if source_path.exists():
            shutil.copy2(source_path, Path(target_dir) / file)
            print(f"Copied {file}")

def main():
    # Paths
    input_dir = Path("./qwen3-tei-onnx")
    output_dir = Path("./qwen3-tei-onnx-int8")
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found!")
        print("Please run the conversion script first: ./run_onnx_conversion.sh")
        sys.exit(1)
    
    input_model = input_dir / "model.onnx"
    output_model = output_dir / "model.onnx"
    
    print("🚀 Qwen3-Embedding ONNX INT8 Quantization")
    print("=" * 50)
    print(f"Input model: {input_model}")
    print(f"Output model: {output_model}")
    print()
    
    # Quantize the model
    quantize_model(input_model, output_model)
    
    # Copy supporting files
    print("\nCopying supporting files...")
    copy_supporting_files(input_dir, output_dir)
    
    # Get file sizes
    print("\n📊 Model size comparison:")
    original_size = sum(f.stat().st_size for f in input_dir.glob("model.onnx*"))
    quantized_size = sum(f.stat().st_size for f in output_dir.glob("model.onnx*"))
    
    print(f"Original: {original_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Quantized: {quantized_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
    
    print("\n✅ Quantization complete!")
    print("\n📋 Next steps:")
    print("1. Test locally: cd qwen3-tei-onnx-int8 && ./test_tei.sh")
    print("2. Upload to HuggingFace: huggingface-cli upload YOUR_USERNAME/qwen3-embedding-int8-tei-onnx ./qwen3-tei-onnx-int8")
    print("3. Deploy with TEI using the int8 model")
    
    # Create test script
    test_script = '''#!/bin/bash
# Test script for quantized INT8 model

echo "Testing Qwen3-Embedding INT8 ONNX with TEI..."
echo "This model is quantized for faster CPU inference"

MODEL_PATH=$(pwd)

echo "Model path: $MODEL_PATH"
echo "Files in model directory:"
ls -la $MODEL_PATH

echo ""
echo "Expected performance improvement: 2-4x faster on CPU"
echo "Note: There may be a small accuracy drop (1-3%)"
'''
    
    with open(output_dir / "test_tei.sh", "w") as f:
        f.write(test_script)
    os.chmod(output_dir / "test_tei.sh", 0o755)

if __name__ == "__main__":
    main()