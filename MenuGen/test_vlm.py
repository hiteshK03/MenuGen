#!/usr/bin/env python3
"""
Simple test script to demonstrate vision-language model integration
Run this to test the new approach vs the old OCR approach
"""

import os
import sys
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def test_florence2_basic():
    """Basic test of Florence-2 model loading and inference"""
    print("🔮 Testing Florence-2 model...")
    
    try:
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", 
            trust_remote_code=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("✅ Model loaded on GPU")
        else:
            print("✅ Model loaded on CPU")
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # Create a simple test image (you can replace this with a real menu image)
        print("Creating test image...")
        test_image = Image.new('RGB', (400, 300), color='white')
        
        # Test OCR functionality
        task_prompt = "<OCR>"
        inputs = processor(text=task_prompt, images=test_image, return_tensors="pt")
        
        # Move inputs to GPU if model is on GPU  
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        print("Running inference...")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3,
                do_sample=False
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print("✅ Inference completed successfully!")
        print(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Florence-2: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'transformers', 
        'PIL',
        'streamlit'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements_vlm.txt")
        return False
    
    return True

def main():
    print("🍽️ MenuGen Vision-Language Model Test")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("💻 CUDA not available, using CPU")
    
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed!")
        return
    
    print()
    
    # Test Florence-2
    if test_florence2_basic():
        print("\n✅ All tests passed! You can now run:")
        print("  streamlit run app_vlm.py")
        print("  or")  
        print("  streamlit run examples/vision_models_demo.py")
    else:
        print("\n❌ Tests failed. Please check your installation.")

if __name__ == "__main__":
    main()
