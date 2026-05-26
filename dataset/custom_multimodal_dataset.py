import torch
import pandas as pd
from PIL import Image
from typing import Optional
from torch.utils.data import Dataset
from src.utils.logger import setup_logger
from qwen_vl_utils import process_vision_info

from dataset.load_dataset import SROIEDataset
from src.vlm_model import load_processor


class SROIEMultimodalDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, split: str, debug: Optional[bool] = False, max_edge: Optional[int]=1024, max_seq_length: Optional[int] = 1024, processor=None):
        self.data_frame = data_frame
        self.split = split
        self.max_edge = max_edge
        self.logger = setup_logger(self.__class__.__name__, debug=debug)
        self.max_seq_length = max_seq_length
        self.processor = processor
        self.is_resized_offline = self.data_frame.attrs.get('is_resized_offline', False)

        if not self.is_resized_offline:
            self.logger.info("Preprocessing resizing has not been done. Resizing of images will happen while loading items")

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        item = self.data_frame.iloc[idx]
        #doc_id = item["doc_id"]
        image_path = item["img_path"]
        entity_path = item["ent_path"]

        # load the iamge
        image = Image.open(image_path).convert("RGB")
        #print(f"Loaded image size: {image.size}")
        # load the label in json string
        with open(entity_path, 'r', encoding='utf-8') as f:
            entities_json_str = f.read().strip()

        if not self.is_resized_offline:
            image.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)

        # construct the prompt and the label for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract company, date, address, and total in JSON format."}
                ]
            },
            # our GT label, as we want to tell the model this is the output we want it to generate
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": entities_json_str}
                ]
            }
        ]

        # lets process so we have proper input tensors for the model
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        #print(f"Processed image inputs shape: {image_inputs[0].size}")
        if len(image_inputs) > 1:
            breakpoint()
        inputs = self.processor(
            text=text,
            images=image_inputs,
            padding="max_length",
            max_length=self.max_seq_length, 
            truncation=True,
            return_tensors="pt"
        )
       
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # advanced optimization (Loss Masking), mask out the prompt tokens
        labels = inputs["input_ids"].clone()
        prompt_text = self.processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        prompt_token_len = len(self.processor.tokenizer(prompt_text)["input_ids"])
        # mask the prompt tokens with -100 so that they are ignored in the loss calculation, only the generation part (the label part) will contribute to the loss
        labels[:prompt_token_len] = -100 
        inputs["labels"] = labels

        #FIX: processor if skips creating mm_token_type_ids, we need to create it ourselves for the multimodal part of the input, which is used in some models to differentiate between text and vision tokens
        if "mm_token_type_ids" not in inputs:
            # 151655 is the standard token ID Qwen2-VL uses to denote vision tokens
            image_token_id = 151655 
            # tensor of zeros matching the shape of input_ids
            mm_token_type_ids = torch.zeros_like(inputs["input_ids"])
            # set positions to 1 wherever there is an image patch token
            mm_token_type_ids[inputs["input_ids"] == image_token_id] = 1
            inputs["mm_token_type_ids"] = mm_token_type_ids
        return inputs
    