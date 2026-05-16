import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


def load_vlm_model():
    # we are using the 2B version of Qwen-VL-Instruct, which is optimized for instruction following and multimodal tasks 
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    # lets load in 4 bit quantization config
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    # now lets load the Processor (Handles Image + Text)
    print(f"Loading Processor and Model for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)

    # model loading with quantization and automatic device mapping
    model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    print("\nModel loaded successfully!")
    print(f"VRAM Footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    return processor, model