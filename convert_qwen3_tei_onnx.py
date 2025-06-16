#!/usr/bin/env python3
"""
Convert Qwen3-Embedding-0.6B to ONNX format specifically for TEI
This script handles the special requirements of TEI's ONNX backend
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from pathlib import Path
import json
import onnx
import onnxruntime as ort

class Qwen3EmbeddingONNXExporter:
    def __init__(self, model_id="Qwen/Qwen3-Embedding-0.6B"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def load_model(self):
        """Load the Qwen3 model and tokenizer"""
        print(f"Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self.model.eval()
        print("Model loaded successfully!")
        
    def create_dummy_inputs(self, batch_size=1, seq_length=128):
        """Create dummy inputs for ONNX export"""
        dummy_text = ["This is a sample text for ONNX export"] * batch_size
        inputs = self.tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_length
        )
        return inputs
    
    def export_to_onnx(self, output_dir="./qwen3-tei-onnx"):
        """Export model to ONNX format compatible with TEI"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer and config
        print("Saving tokenizer and config...")
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        
        # Create config_sentence_transformers.json for TEI
        st_config = {
            "max_seq_length": 32768,  # Qwen3's max length
            "do_lower_case": False,
            "model_type": "qwen3",
            "tokenizer_type": "qwen3",
            "normalize": False,  # TEI will handle normalization
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,  # Mean pooling as specified in porter.yaml
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False
        }
        
        with open(output_path / "config_sentence_transformers.json", "w") as f:
            json.dump(st_config, f, indent=2)
        
        # Create dummy inputs
        dummy_inputs = self.create_dummy_inputs()
        input_ids = dummy_inputs["input_ids"]
        attention_mask = dummy_inputs["attention_mask"]
        
        # Define the forward function for ONNX export
        def forward_for_export(input_ids, attention_mask):
            # Get model outputs
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extract hidden states (last layer embeddings)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state
            else:
                # For Qwen3, it might be in a different attribute
                embeddings = outputs[0]
            
            # Apply mean pooling (as specified in porter.yaml)
            # Expand attention mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            return embeddings
        
        # Export to ONNX
        print("Exporting to ONNX...")
        onnx_path = output_path / "model.onnx"
        
        # Wrap the model to include pooling
        class ModelWithPooling(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                else:
                    embeddings = outputs[0]
                
                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                return embeddings
        
        wrapped_model = ModelWithPooling(self.model)
        wrapped_model.eval()
        
        # Export with external data in single file for TEI compatibility
        with torch.no_grad():
            # First export to a temporary file
            temp_onnx = output_path / "temp_model.onnx"
            torch.onnx.export(
                wrapped_model,
                (input_ids, attention_mask),
                str(temp_onnx),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'last_hidden_state': {0: 'batch_size'}  # Only batch dimension is dynamic
                },
                verbose=False
            )
        
        print("Consolidating external data for TEI...")
        # Load the model (this loads all external data)
        onnx_model = onnx.load(str(temp_onnx))
        
        # Remove temporary files
        temp_onnx.unlink()
        for file in output_path.glob("onnx__*"):
            file.unlink()
        if (output_path / "model.embed_tokens.weight").exists():
            (output_path / "model.embed_tokens.weight").unlink()
        
        # Save with consolidated external data
        onnx.save_model(
            onnx_model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx_data",  # TEI expects this exact filename
            size_threshold=1024,  # Externalize tensors larger than 1KB
            convert_attribute=False
        )
        
        print(f"ONNX model saved to: {onnx_path}")
        print(f"External data saved to: {output_path / 'model.onnx_data'}")
        
        # Verify the ONNX model
        print("Verifying ONNX model...")
        # For large models, we need to check with the file path
        onnx.checker.check_model(str(onnx_path))
        print("ONNX model verification passed!")
        
        # Test with ONNX Runtime
        print("Testing with ONNX Runtime...")
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # Prepare inputs
        ort_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy()
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"ONNX output shape: {ort_outputs[0].shape}")
        print(f"Expected shape: [batch_size, embedding_dim]")
        
        # Create a simple test script
        test_script = '''#!/bin/bash
# Test script for TEI with the converted ONNX model

echo "Testing Qwen3-Embedding-0.6B ONNX with TEI..."

# Update porter.yaml to use local model
MODEL_PATH=$(pwd)/qwen3-tei-onnx

echo "Model path: $MODEL_PATH"
echo "Files in model directory:"
ls -la $MODEL_PATH

echo ""
echo "To use this model with TEI:"
echo "1. Upload to HuggingFace Hub, or"
echo "2. Mount this directory in your TEI container"
echo "3. Update model-id in porter.yaml to point to this model"
'''
        
        with open(output_path / "test_tei.sh", "w") as f:
            f.write(test_script)
        os.chmod(output_path / "test_tei.sh", 0o755)
        
        print(f"\n✅ Conversion complete!")
        print(f"📁 Output directory: {output_dir}")
        print("\n📋 Next steps:")
        print("1. Upload to HuggingFace Hub: huggingface-cli upload YOUR_USERNAME/qwen3-embedding-onnx ./qwen3-tei-onnx")
        print("2. Or use locally by mounting the directory in TEI")
        print("3. Update porter.yaml with the new model-id")
        
        # List files
        print("\n📄 Created files:")
        for file in sorted(output_path.rglob("*")):
            if file.is_file():
                print(f"  - {file.relative_to(output_path)}")

def main():
    exporter = Qwen3EmbeddingONNXExporter()
    exporter.load_model()
    exporter.export_to_onnx()

if __name__ == "__main__":
    # Check dependencies
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("Please install required packages:")
        print("pip install torch transformers onnx onnxruntime")
        sys.exit(1)
    
    main()