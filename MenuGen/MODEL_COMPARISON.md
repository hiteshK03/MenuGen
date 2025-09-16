# Vision-Language Models for MenuGen

This document compares different open-source models available on Hugging Face for replacing the OCR-based menu text extraction with direct image understanding.

## Recommended Models

### 🥇 **SmolVLM2** (CHOSEN IMPLEMENTATION ✅)
- **Model**: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- **Strengths**: 
  - Ultra-efficient with excellent performance
  - Natural language understanding of images
  - Great instruction following for specific tasks
  - Optimized for low resource usage
- **Use Case**: Primary implementation in MenuGen
- **GPU Memory**: <4GB
- **Integration**: ✅ **IMPLEMENTED** in `app_vlm.py`

### 🥈 **Microsoft Florence-2** (Alternative)
- **Models**: `microsoft/Florence-2-base` (232M), `microsoft/Florence-2-large` (771M)
- **Strengths**: 
  - Excellent OCR-free document understanding
  - Understands spatial layout and structure
  - Can extract text with bounding boxes
  - Good performance on menu-like documents
- **Use Case**: Alternative implementation option
- **GPU Memory**: ~2GB for base, ~4GB for large
- **Integration**: 🔄 Available as alternative

### 🚀 **SmolVLM** (Most Efficient)
- **Model**: `HuggingFaceTB/SmolVLM-Instruct`
- **Strengths**: 
  - Extremely efficient (<1GB GPU memory)
  - Fast inference
  - Good instruction following
- **Use Case**: Resource-constrained environments
- **GPU Memory**: <1GB
- **Integration**: ✅ Available in demo

### 🎯 **Qwen2-VL** (Multilingual)
- **Models**: `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`
- **Strengths**: 
  - Excellent multilingual support
  - Strong document understanding
  - Good reasoning capabilities
- **Use Case**: International restaurants, complex layouts
- **GPU Memory**: 4GB (2B), 14GB (7B)
- **Integration**: 🔄 Can be integrated

### 📄 **Microsoft TrOCR** (Text-Focused)
- **Model**: `microsoft/trocr-base-printed`
- **Strengths**: 
  - Specialized for text recognition
  - End-to-end text extraction
  - Good for clean printed text
- **Use Case**: Simple text extraction only
- **GPU Memory**: ~2GB
- **Integration**: ✅ Available in demo

### 🔍 **Donut** (Structured Documents)
- **Model**: `naver-clova-ix/donut-base-finetuned-docvqa`
- **Strengths**: 
  - Designed for document understanding
  - Can answer questions about content
  - Good for structured information
- **Use Case**: Q&A about menu content
- **GPU Memory**: ~1.5GB
- **Integration**: 🔄 Can be integrated

## Performance Comparison

| Model | Size | GPU Memory | Speed | Menu Accuracy | Best For | Status |
|-------|------|------------|--------|---------------|-----------|--------|
| **SmolVLM2-2.2B** | **2.2B** | **<4GB** | **Very Fast** | **⭐⭐⭐⭐⭐** | **Menu understanding** | **✅ IMPLEMENTED** |
| SmolVLM | 1.7B | <1GB | Very Fast | ⭐⭐⭐⭐ | Resource efficiency | 🔄 Available |
| Florence-2-base | 232M | ~2GB | Fast | ⭐⭐⭐⭐⭐ | General menu understanding | 🔄 Alternative |
| Florence-2-large | 771M | ~4GB | Medium | ⭐⭐⭐⭐⭐ | Complex layouts | 🔄 Alternative |
| Qwen2-VL-2B | 2B | ~4GB | Fast | ⭐⭐⭐⭐⭐ | Multilingual menus | 🔄 Available |
| TrOCR | 558M | ~2GB | Fast | ⭐⭐⭐ | Simple text extraction | 🔄 Available |
| Donut | 200M | ~1.5GB | Medium | ⭐⭐⭐⭐ | Structured Q&A | 🔄 Available |

## Implementation Status

### ✅ **Completed**
- Research and comparison of models
- **SmolVLM2 integration in `app_vlm.py`** (PRIMARY IMPLEMENTATION)
- Demo application with multiple models
- Updated requirements file for SmolVLM2
- Created dedicated SmolVLM2 test script (`test_smolvlm.py`)

### 🔄 **Available for Integration**
- Florence-2 example code (alternative approach)
- SmolVLM example code (smaller version)
- TrOCR example code
- BLIP-2 example code

### 📋 **Next Steps**
- Test SmolVLM2 with various menu types
- Performance optimization and benchmarking
- User feedback and fine-tuning

## Code Examples

### SmolVLM2 (IMPLEMENTED ✅)
```python
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Extract menu items with natural language prompt
prompt = "Please analyze this restaurant menu image and extract all the dish names."
inputs = processor(images=image, text=prompt, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=500)
result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Florence-2 (Alternative)
```python
from transformers import AutoProcessor, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base", 
    torch_dtype=torch.float16,
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", 
    trust_remote_code=True
)

# Extract text with regions
task_prompt = "<OCR_WITH_REGION>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
result = processor.post_process_generation(generated_text, task=task_prompt)
```

### SmolVLM (Efficient Alternative)
```python
from transformers import pipeline

pipe = pipeline("image-to-text", model="HuggingFaceTB/SmolVLM-Instruct")
messages = [{
    "role": "user", 
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "List all dish names from this menu."}
    ]
}]
result = pipe(messages, max_new_tokens=500)
```

## Migration from OCR

### Before (OCR-based)
```python
# Old approach with pytesseract
import pytesseract
text = pytesseract.image_to_string(image)
dish_name = text.split('\n')[0].strip()
```

### After (Vision-Language Model)
```python
# New approach with Florence-2
model, processor = load_menu_understanding_model()
menu_items = extract_menu_items(image, model, processor)
# Get structured list of all dishes, not just first line
```

### Benefits of Migration
- ✅ **Better Accuracy**: Understands context, not just character recognition
- ✅ **Structured Output**: Gets list of dishes, prices, descriptions
- ✅ **Layout Understanding**: Recognizes menu structure and sections
- ✅ **Robustness**: Works with various image qualities and formats
- ✅ **No Preprocessing**: No need for image enhancement or OCR tuning

## Hardware Requirements

### Minimum (CPU Only)
- RAM: 8GB
- Model: SmolVLM or TrOCR-base
- Performance: Slow but functional

### Recommended (GPU)
- GPU: 4GB VRAM (RTX 3060 or better)
- Model: Florence-2-base or Qwen2-VL-2B
- Performance: Fast inference

### High-Performance (Large GPU)
- GPU: 8GB+ VRAM (RTX 4070 or better)
- Model: Florence-2-large or Qwen2-VL-7B
- Performance: Best accuracy and speed
