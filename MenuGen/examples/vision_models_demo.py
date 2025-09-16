"""
Vision-Language Models Demo for MenuGen
Compare different approaches for extracting dishes from menu images
"""

import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForCausalLM,  # For Florence-2
    BlipProcessor, BlipForConditionalGeneration,  # For BLIP-2
    TrOCRProcessor, VisionEncoderDecoderModel,  # For TrOCR
    DonutProcessor, VisionEncoderDecoderModel as DonutModel,  # For Donut
    pipeline  # For SmolVLM
)

# Florence-2 - Best Overall Choice
@st.cache_resource
def load_florence2():
    """Load Microsoft Florence-2 for document understanding"""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", 
            trust_remote_code=True
        )
        return model, processor
    except Exception as e:
        st.error(f"Failed to load Florence-2: {e}")
        return None, None

def extract_dishes_florence2(image, model, processor):
    """Extract dishes using Florence-2"""
    if model is None or processor is None:
        return "Model not loaded"
    
    # Task: OCR with region
    task_prompt = "<OCR_WITH_REGION>"
    
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

# SmolVLM - Most Efficient Choice
@st.cache_resource  
def load_smolvlm():
    """Load SmolVLM for efficient processing"""
    try:
        pipe = pipeline(
            "image-to-text", 
            model="HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.float16
        )
        return pipe
    except Exception as e:
        st.error(f"Failed to load SmolVLM: {e}")
        return None

def extract_dishes_smolvlm(image, pipe):
    """Extract dishes using SmolVLM with custom prompt"""
    if pipe is None:
        return "Model not loaded"
    
    # Custom prompt for dish extraction
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "List all the dish names visible in this menu image, one per line."}
            ]
        }
    ]
    
    result = pipe(messages, max_new_tokens=500)
    return result[0]['generated_text']

# TrOCR - Specialized Text Recognition
@st.cache_resource
def load_trocr():
    """Load TrOCR for text recognition"""
    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        return model, processor
    except Exception as e:
        st.error(f"Failed to load TrOCR: {e}")
        return None, None

def extract_text_trocr(image, model, processor):
    """Extract text using TrOCR"""
    if model is None or processor is None:
        return "Model not loaded"
    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# BLIP-2 - Good General Purpose
@st.cache_resource
def load_blip2():
    """Load BLIP-2 for image captioning"""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        return model, processor
    except Exception as e:
        st.error(f"Failed to load BLIP-2: {e}")
        return None, None

def extract_dishes_blip2(image, model, processor):
    """Extract dishes using BLIP-2 with custom prompt"""
    if model is None or processor is None:
        return "Model not loaded"
    
    # Conditional generation with custom prompt
    text = "menu with dishes:"
    inputs = processor(image, text, return_tensors="pt")
    
    out = model.generate(**inputs, max_length=150)
    generated_text = processor.decode(out[0], skip_special_tokens=True)
    return generated_text

# Streamlit Demo Interface
def main():
    st.title("Vision Models Comparison for MenuGen")
    st.write("Compare different vision-language models for extracting dishes from menu images")
    
    # Model selection
    model_choice = st.selectbox(
        "Choose a model to test:",
        ["Florence-2 (Recommended)", "SmolVLM (Efficient)", "TrOCR (Text Focus)", "BLIP-2 (General)"]
    )
    
    uploaded_file = st.file_uploader("Upload a menu image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Menu Image", use_column_width=True)
        
        if st.button("Extract Dishes"):
            if "Florence-2" in model_choice:
                with st.spinner("Loading Florence-2..."):
                    model, processor = load_florence2()
                with st.spinner("Extracting dishes with Florence-2..."):
                    result = extract_dishes_florence2(image, model, processor)
                    st.json(result)
                    
            elif "SmolVLM" in model_choice:
                with st.spinner("Loading SmolVLM..."):
                    pipe = load_smolvlm()
                with st.spinner("Extracting dishes with SmolVLM..."):
                    result = extract_dishes_smolvlm(image, pipe)
                    st.text_area("Extracted Dishes", result, height=200)
                    
            elif "TrOCR" in model_choice:
                with st.spinner("Loading TrOCR..."):
                    model, processor = load_trocr()
                with st.spinner("Extracting text with TrOCR..."):
                    result = extract_text_trocr(image, model, processor)
                    st.text_area("Extracted Text", result, height=200)
                    
            elif "BLIP-2" in model_choice:
                with st.spinner("Loading BLIP-2..."):
                    model, processor = load_blip2()
                with st.spinner("Analyzing image with BLIP-2..."):
                    result = extract_dishes_blip2(image, model, processor)
                    st.text_area("Generated Caption", result, height=200)

if __name__ == "__main__":
    main()
