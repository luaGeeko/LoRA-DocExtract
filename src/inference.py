import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

def run_vlm_inference_per_sample(dataframe_row, model, processor):
    image_path = dataframe_row["img_path"]
    entity_path = dataframe_row["ent_path"]
    
    # raw data components
    image = Image.open(image_path).convert("RGB")
    with open(entity_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read().strip()
        
    # prompt template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract company, date, address, and total in JSON format."}
            ]
        }
    ]
    
    # process inputs exactly like training
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=False, # No padding needed for single-item inference
        return_tensors="pt"
    )
    
    # move tensors to the T4 GPU
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # set model in evaluation mode and generate output
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        # trim the prompt tokens out of the generation length
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
    print("\n" + "="*50)
    print(f"DOCUMENT ID: {dataframe_row['doc_id']}")
    print("="*50)
    print(f"GROUND TRUTH:\n{ground_truth}")
    print("-"*50)
    print(f"LORA PREDICTION:\n{output_text}")
    print("="*50 + "\n")
    
    return ground_truth, output_text
