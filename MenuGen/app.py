import streamlit as st
from PIL import Image
import pytesseract
import requests
from transformers import pipeline
from diffusers import FluxPipeline
import torch
import numpy as np
import cv2

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image):
    open_cv_image = np.array(image)
    # Convert RGB to BGR 
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    text = pytesseract.image_to_string(img)
    return text

# Function to get ingredients using a web search (simulated)
def get_ingredients(dish_name):
    pipe = pipeline("text-generation", model="flax-community/t5-recipe-generation")
    ingredients = pipe(dish_name)
    return ingredients

# Function to generate an image of the dish
def generate_dish_image(dish_name):
    # Load a pre-trained Stable Diffusion model
    model_id = "black-forest-labs/FLUX.1-schnell"
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    prompt = f"A high-quality photograph of {dish_name}"
    image = pipe(prompt).images[0]
    return image

st.title("MenuGen: From Menu to Masterpiece")

st.write("Upload an image of a restaurant menu to get started.")

uploaded_file = st.file_uploader("Choose a menu image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Menu", use_column_width=True)

    if st.button("Process Menu"):
        with st.spinner("Extracting text from the menu..."):
            menu_text = extract_text_from_image(image)
            st.text_area("Extracted Text", menu_text, height=200)

        # For simplicity, we'll just use the first line of the extracted text as the dish name
        # In a real app, you would need more sophisticated logic to parse the menu
        dish_name = menu_text.split('\n')[0].strip()

        if dish_name:
            st.subheader(f"Dish: {dish_name}")

            with st.spinner(f"Searching for ingredients for {dish_name}..."):
                ingredients = get_ingredients(dish_name)
                st.write(ingredients)

            with st.spinner(f"Generating an image of {dish_name}..."):
                dish_image = generate_dish_image(dish_name)
                st.image(dish_image, caption=f"Generated image of {dish_name}", use_column_width=True)
        else:
            st.warning("Could not identify a dish from the menu.")
