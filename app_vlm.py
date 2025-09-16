"""
MenuGen: From Menu to Masterpiece (SmolVLM Version)
Enhanced with SmolVLM2 for direct image understanding - no OCR intermediate step
"""

import streamlit as st
import torch
from PIL import Image
import re
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    pipeline
)
from diffusers import FluxPipeline

# Vision-Language Model for Menu Understanding
@st.cache_resource
def load_menu_understanding_model():
    """Load SmolVLM2 for direct menu understanding"""
    try:
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        return model, processor
    except Exception as e:
        st.error(f"Failed to load SmolVLM2 model: {e}")
        return None, None

def extract_menu_items(image, model, processor):
    """
    Extract menu items directly from image using SmolVLM2
    Returns list of dish names found in the menu
    """
    if model is None or processor is None:
        return []
    
    try:
        # Use proper SmolVLM2 chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Please analyze this restaurant menu image and extract all the dish names. List only the dish names, one per line, without prices or descriptions. Focus on the main dishes, appetizers, and entrees."}
                ]
            }
        ]
        
        # Apply chat template and process
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        
        # Move inputs to the same device as model
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=500,
                do_sample=False,  # Use greedy decoding for consistency
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        input_length = inputs['input_ids'].shape[1]
        generated_text = processor.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
        
        # Debug: Show raw model output if in debug mode
        if st.sidebar.checkbox("🔍 Debug Mode", help="Show raw model output"):
            st.sidebar.text_area("Raw SmolVLM2 Response", generated_text, height=100)
        
        # Parse the response to extract dish names
        return parse_dish_names_from_response(generated_text)
        
    except Exception as e:
        st.error(f"Error extracting menu items: {e}")
        return []

def parse_dish_names_from_response(response_text):
    """
    Parse SmolVLM2's natural language response to extract dish names
    Handles both line-separated and comma-separated dish lists
    """
    dishes = []
    
    # Split response into lines
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, lines with just numbers, or very short lines
        if not line or line.isdigit() or len(line) < 3:
            continue
            
        # Remove common prefixes like "1. ", "- ", "• ", etc.
        clean_line = re.sub(r'^[\d\.\-•\*\s]*', '', line).strip()
        
        # Skip lines that look like prices (contain $ € £ ¥ ₹ or pure numbers)
        if re.search(r'[$€£¥₹]|\b\d+[\.,]\d{2}\b$', clean_line):
            continue
            
        # Skip very short items after cleaning
        if len(clean_line) < 3:
            continue
            
        # Remove any trailing prices from the line
        clean_line = re.sub(r'\s*[-–]\s*[$€£¥₹][\d\.,]+.*$', '', clean_line)
        clean_line = re.sub(r'\s*[$€£¥₹][\d\.,]+.*$', '', clean_line)
        
        # Check if this line contains multiple comma-separated dishes
        if ',' in clean_line and len(clean_line) > 50:  # Likely comma-separated list
            # Split by commas and process each dish
            comma_dishes = clean_line.split(',')
            for dish in comma_dishes:
                dish = dish.strip()
                # Remove any remaining prefixes from individual dishes
                dish = re.sub(r'^[\d\.\-•\*\s]*', '', dish).strip()
                # Remove parenthetical descriptions if they're at the end
                dish = re.sub(r'\s*\([^)]*\)$', '', dish).strip()
                
                if dish and len(dish) > 2:
                    dishes.append(dish)
        else:
            # Single dish per line
            if clean_line and len(clean_line) > 2:
                dishes.append(clean_line.strip())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dishes = []
    for dish in dishes:
        dish_clean = dish.strip()
        if dish_clean and dish_clean.lower() not in seen and len(dish_clean) > 2:
            seen.add(dish_clean.lower())
            unique_dishes.append(dish_clean)
    
    return unique_dishes[:15]  # Limit to 15 dishes to avoid overwhelming the UI

# Image generation functions only

@st.cache_resource
def load_image_generator():
    """Load image generation model"""
    try:
        model_id = "black-forest-labs/FLUX.1-schnell"
        pipe = FluxPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe
    except Exception as e:
        st.error(f"Failed to load image generation model: {e}")
        return None

def generate_dish_image(dish_name, pipe):
    """Generate an image of the dish"""
    if pipe is None:
        return None
        
    try:
        print(f"Generating image for {dish_name}")
        prompt = f"A high-quality professional photograph of {dish_name}, food photography, appetizing"
        image = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Main Streamlit App
def main():
    st.title("🍽️ MenuGen: From Menu to Masterpiece")
    st.markdown("**Powered by SmolVLM2** - Ultra-efficient AI vision for menu understanding!")
    
    st.write("Upload an image of a restaurant menu to get started.")
    
    uploaded_file = st.file_uploader(
        "Choose a menu image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of a restaurant menu for best results"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📋 Uploaded Menu", use_column_width=True)
        
        # Model loading status
        with st.status("Loading AI models...") as status:
            st.write("🔮 Loading SmolVLM2 menu understanding model...")
            menu_model, menu_processor = load_menu_understanding_model()
            
            st.write("🎨 Loading image generator...")
            image_pipe = load_image_generator()
            
            status.update(label="✅ Models loaded successfully!", state="complete")
        
        # Automatically start processing after models are loaded
        # Extract menu items using vision model
        with st.spinner("🔍 Analyzing menu with AI vision..."):
            menu_items = extract_menu_items(image, menu_model, menu_processor)
            print(f"Menu items: {menu_items}")
        
        if menu_items:
            st.success(f"Found {len(menu_items)} menu items!")
            
            # Display found items
            with st.expander("📋 Detected Menu Items", expanded=True):
                for i, item in enumerate(menu_items, 1):
                    st.write(f"{i}. {item}")
            
            # Automatically generate images for all dishes
            st.subheader("🎨 Generated Images for All Dishes")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, dish_name in enumerate(menu_items):
                # Update progress
                progress = (idx + 1) / len(menu_items)
                progress_bar.progress(progress)
                status_text.text(f"Processing {dish_name} ({idx + 1}/{len(menu_items)})...")
                
                # Create expandable section for each dish
                with st.expander(f"🍽️ {dish_name}", expanded=True):
                    # Generate image only
                    st.subheader("📸 Generated Image")
                    with st.spinner(f"Creating image..."):
                        dish_image = generate_dish_image(dish_name, image_pipe)
                        if dish_image:
                            st.image(
                                dish_image, 
                                caption=f"AI-generated image of {dish_name}",
                                use_column_width=True
                            )
                        else:
                            st.warning("Could not generate image for this dish")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Generated images for all {len(menu_items)} dishes!")
        else:
            st.warning("🤔 Could not identify any dishes from the menu. Please try with a clearer image.")
            st.info("💡 Tips: Ensure the menu text is clearly visible and well-lit")

    # Sidebar with information
    with st.sidebar:
        st.markdown("### 🤖 AI Models Used")
        st.markdown("- **Menu Understanding**: SmolVLM2 (2.2B params)")
        st.markdown("- **Image Generation**: FLUX.1")
        
        st.markdown("### ✨ Features")
        st.markdown("- 🔍 Ultra-efficient vision-language understanding")
        st.markdown("- 🎨 AI-generated dish images")
        st.markdown("- ⚡ Automatic processing (no button clicks needed)")
        st.markdown("- ⚡ Optimized for low GPU memory usage")
        st.markdown("- 🚀 Fast inference with SmolVLM2")
        
        # System info
        if torch.cuda.is_available():
            st.success("🚀 GPU acceleration enabled")
            gpu_name = torch.cuda.get_device_name(0)
            st.info(f"GPU: {gpu_name}")
        else:
            st.info("💻 Running on CPU")
            
        # Model info
        st.markdown("### 📊 Model Info")
        st.markdown("- **Size**: ~2.2B parameters")
        st.markdown("- **Memory**: <4GB GPU RAM")
        st.markdown("- **Speed**: Ultra-fast inference")

if __name__ == "__main__":
    main()
