"""
MenuGen: From Menu to Masterpiece (Gradio Version)
Enhanced with SmolVLM2 for direct image understanding - no OCR intermediate step
"""

import gradio as gr
import torch
from PIL import Image
import re
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    pipeline
)
from diffusers import FluxPipeline
import functools

# Global variables to store loaded models
menu_model = None
menu_processor = None  
image_pipe = None
debug_mode = False

def load_menu_understanding_model():
    """Load SmolVLM2 for direct menu understanding"""
    global menu_model, menu_processor
    if menu_model is not None and menu_processor is not None:
        return menu_model, menu_processor
    
    try:
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        menu_model, menu_processor = model, processor
        return model, processor
    except Exception as e:
        print(f"Failed to load SmolVLM2 model: {e}")
        return None, None

def load_image_generator():
    """Load image generation model"""
    global image_pipe
    if image_pipe is not None:
        return image_pipe
        
    try:
        model_id = "black-forest-labs/FLUX.1-schnell"
        pipe = FluxPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        image_pipe = pipe
        return pipe
    except Exception as e:
        print(f"Failed to load image generation model: {e}")
        return None

def extract_menu_items(image, model, processor, debug=False):
    """
    Extract menu items directly from image using SmolVLM2
    Returns list of dish names found in the menu and debug info
    """
    if model is None or processor is None:
        print("Model or processor is None")
        return [], ""
    
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
            # Only convert specific tensors to bfloat16, keep token IDs as integers
            inputs = {
                k: (v.to(model.device, dtype=torch.bfloat16) if torch.is_tensor(v) and k == 'pixel_values' 
                   else v.to(model.device) if torch.is_tensor(v) 
                   else v) 
                for k, v in inputs.items()
            }
        
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
        
        # Parse the response to extract dish names
        dishes = parse_dish_names_from_response(generated_text)
        debug_info = generated_text if debug else ""
        
        return dishes, debug_info
        
    except Exception as e:
        error_msg = f"Error extracting menu items: {e}"
        print(error_msg)
        return [], error_msg

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
        print(f"Error generating image: {e}")
        return None

def process_menu(image, debug_mode, progress=gr.Progress()):
    """Main processing function for the Gradio interface"""
    if image is None:
        return None, "Please upload a menu image first.", "", []
    
    # Load models
    progress(0.1, desc="Loading SmolVLM2 menu understanding model...")
    menu_model, menu_processor = load_menu_understanding_model()
    
    if menu_model is None:
        return None, "Failed to load menu understanding model.", "", []
    
    progress(0.3, desc="Loading image generator...")
    image_pipe = load_image_generator()
    
    if image_pipe is None:
        return None, "Failed to load image generation model.", "", []
    
    progress(0.5, desc="Analyzing menu with AI vision...")
    
    # Extract menu items
    menu_items, debug_info = extract_menu_items(image, menu_model, menu_processor, debug_mode)
    
    if not menu_items:
        return None, "🤔 Could not identify any dishes from the menu. Please try with a clearer image.\n💡 Tips: Ensure the menu text is clearly visible and well-lit", debug_info, []
    
    # Generate images for all dishes
    generated_images = []
    status_message = f"Found {len(menu_items)} menu items!\n\n📋 Detected Menu Items:\n"
    
    for i, item in enumerate(menu_items, 1):
        status_message += f"{i}. {item}\n"
    
    status_message += "\n🎨 Generating images for dishes...\n"
    
    for idx, dish_name in enumerate(menu_items):
        progress_val = 0.5 + (0.5 * (idx + 1) / len(menu_items))
        progress(progress_val, desc=f"Generating image for {dish_name} ({idx + 1}/{len(menu_items)})...")
        
        dish_image = generate_dish_image(dish_name, image_pipe)
        if dish_image:
            generated_images.append((dish_image, f"🍽️ {dish_name}"))
        else:
            # Add a placeholder for failed generations
            generated_images.append((None, f"🍽️ {dish_name} (generation failed)"))
    
    final_status = f"✅ Successfully processed {len(menu_items)} dishes!\n\n📋 Menu Items Found:\n"
    for i, item in enumerate(menu_items, 1):
        final_status += f"{i}. {item}\n"
    
    return image, final_status, debug_info, generated_images

def get_system_info():
    """Get system information for display"""
    info = []
    info.append("🤖 **AI Models Used:**")
    info.append("- Menu Understanding: SmolVLM2 (2.2B params)")
    info.append("- Image Generation: FLUX.1")
    info.append("")
    info.append("✨ **Features:**")
    info.append("- 🔍 Ultra-efficient vision-language understanding")
    info.append("- 🎨 AI-generated dish images")
    info.append("- ⚡ Optimized for low GPU memory usage")
    info.append("- 🚀 Fast inference with SmolVLM2")
    info.append("")
    
    if torch.cuda.is_available():
        info.append("🚀 **GPU acceleration enabled**")
        gpu_name = torch.cuda.get_device_name(0)
        info.append(f"GPU: {gpu_name}")
    else:
        info.append("💻 **Running on CPU**")
    
    info.append("")
    info.append("📊 **Model Info:**")
    info.append("- Size: ~2.2B parameters")
    info.append("- Memory: <4GB GPU RAM")
    info.append("- Speed: Ultra-fast inference")
    
    return "\n".join(info)

# Create the Gradio interface
def create_interface():
    with gr.Blocks(
        title="MenuGen: From Menu to Masterpiece",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🍽️ MenuGen: From Menu to Masterpiece
            **Powered by SmolVLM2** - Ultra-efficient AI vision for menu understanding!
            
            Upload an image of a restaurant menu to get started.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                image_input = gr.Image(
                    label="📋 Upload Menu Image",
                    type="pil",
                    height=400
                )
                
                debug_checkbox = gr.Checkbox(
                    label="🔍 Debug Mode",
                    info="Show raw SmolVLM2 response",
                    value=False
                )
                
                process_btn = gr.Button(
                    "🚀 Process Menu",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # System info
                system_info = gr.Markdown(
                    get_system_info(),
                    label="System Information"
                )
        
        # Output sections
        with gr.Row():
            with gr.Column():
                # Status and results
                status_output = gr.Textbox(
                    label="📊 Processing Status",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                # Debug output (only shown when debug mode is on)
                debug_output = gr.Textbox(
                    label="🔍 Raw SmolVLM2 Response (Debug)",
                    lines=5,
                    max_lines=15,
                    interactive=False,
                    visible=False
                )
        
        # Generated images gallery
        generated_gallery = gr.Gallery(
            label="🎨 Generated Dish Images",
            show_label=True,
            elem_id="gallery",
            columns=3,
            rows=3,
            height="auto",
            object_fit="cover"
        )
        
        # Event handlers
        def toggle_debug(debug_enabled):
            return gr.update(visible=debug_enabled)
        
        debug_checkbox.change(
            toggle_debug,
            inputs=[debug_checkbox],
            outputs=[debug_output]
        )
        
        process_btn.click(
            fn=process_menu,
            inputs=[image_input, debug_checkbox],
            outputs=[image_input, status_output, debug_output, generated_gallery]
        )
    
    return demo

# if __name__ == "__main__":
#     demo = create_interface()
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         share=True,
#         show_error=True
#     )
