#!/usr/bin/env python3
"""
SmolVLM2 Test Script for MenuGen
Test the SmolVLM2 model integration for menu understanding
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def create_test_menu_image():
    """Create a simple test menu image for testing"""
    # Create a simple menu image
    img = Image.new('RGB', (600, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_item = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_title = ImageFont.load_default()
        font_item = ImageFont.load_default()
    
    # Draw menu title
    draw.text((50, 30), "RESTAURANT MENU", fill='black', font=font_title)
    
    # Draw menu items
    menu_items = [
        "APPETIZERS",
        "Caesar Salad - $12",
        "Buffalo Wings - $15",
        "Mozzarella Sticks - $10",
        "",
        "MAIN DISHES", 
        "Grilled Salmon - $28",
        "Beef Burger - $18",
        "Chicken Pasta - $22",
        "Vegetarian Pizza - $20",
        "",
        "DESSERTS",
        "Chocolate Cake - $8",
        "Ice Cream Sundae - $6"
    ]
    
    y_pos = 80
    for item in menu_items:
        if item == "":
            y_pos += 10
        elif item.isupper() and "-" not in item:  # Section headers
            draw.text((50, y_pos), item, fill='black', font=font_title)
            y_pos += 40
        else:
            draw.text((70, y_pos), item, fill='black', font=font_item)
            y_pos += 25
    
    return img

def test_smolvlm2():
    """Test SmolVLM2 model loading and inference"""
    print("🔮 Testing SmolVLM2 model...")
    
    try:
        print("Loading model and processor...")
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        if torch.cuda.is_available() and hasattr(model, 'device'):
            print(f"Model device: {model.device}")
        
        # Create test image
        print("Creating test menu image...")
        test_image = create_test_menu_image()
        test_image.save("test_menu.png")
        print("✅ Test menu image saved as 'test_menu.png'")
        
        # Test menu analysis using SmolVLM2 chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": "test_menu.png"},
                    {"type": "text", "text": "Please analyze this restaurant menu image and extract all the dish names. List only the dish names, one per line, without prices or descriptions."}
                ]
            }
        ]
        
        print("Processing image with SmolVLM2...")
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[test_image], return_tensors="pt")
        
        # Move inputs to the same device as model if needed
        if hasattr(model, 'device') and torch.cuda.is_available():
            inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=300,
                do_sample=False,  # Use greedy decoding for more consistent results
                temperature=None,
                top_p=None,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (not the input)
        input_length = inputs['input_ids'].shape[1]
        generated_text = processor.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
        
        print("\n" + "="*50)
        print("📋 SMOLVLM2 RESPONSE:")
        print("="*50)
        print(f"Raw response: '{generated_text}'")
        print("="*50)
        
        # Also show the full generated text for debugging
        full_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("\n🔍 FULL GENERATED TEXT (for debugging):")
        print("-" * 50)
        print(f"Full: '{full_text}'")
        print("-" * 50)
        
        print(f"\n📊 Generation Stats:")
        print(f"Input tokens: {input_length}")
        print(f"Total tokens: {generated_ids.shape[1]}")
        print(f"Generated tokens: {generated_ids.shape[1] - input_length}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing SmolVLM2: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'), 
        ('PIL', 'Pillow'),
        ('streamlit', 'streamlit')
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"❌ {package_name}")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    # Check transformers version
    try:
        import transformers
        version = transformers.__version__
        print(f"📦 Transformers version: {version}")
        
        # Check if version is recent enough for SmolVLM2
        major, minor = map(int, version.split('.')[:2])
        if major < 4 or (major == 4 and minor < 35):
            print("⚠️  Transformers version may be too old. Recommended: >=4.35.0")
    except:
        pass
    
    return True

def main():
    print("🍽️ SmolVLM2 MenuGen Test")
    print("=" * 50)
    print("DEBUG: Script is running")
    
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
    
    # Test SmolVLM2
    if test_smolvlm2():
        print("\n✅ All tests passed! SmolVLM2 is working correctly.")
        print("You can now run:")
        print("  streamlit run app_vlm.py")
        
        # Clean up test image
        try:
            os.remove("test_menu.png")
            print("🗑️  Cleaned up test image")
        except:
            pass
    else:
        print("\n❌ Tests failed. Please check your installation.")

if __name__ == "__main__":
    main()
