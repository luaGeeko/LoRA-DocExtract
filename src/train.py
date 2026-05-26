import torch
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer
from src.vlm_model import load_vlm_model, load_processor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Importing the dataset classes we perfected earlier
from dataset.load_dataset import SROIEDataset
from dataset.custom_multimodal_dataset import  SROIEMultimodalDataset

def vlm_collate_fn(batch):
    """
    Custom collator for Vision-Language Models.
    Safely stacks text tensors and concatenates dynamic visual patch tensors.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        # Vision tensors vary in sequence length depending on the image size, 
        # so they must be concatenated along the 0-th dimension, not stacked.
        # CRITICAL ADDITION: Stack the multimodal token type mappings
        "mm_token_type_ids": torch.stack([item["mm_token_type_ids"] for item in batch]),
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.stack([item["image_grid_thw"] for item in batch], dim=0)
    }

def train_model(cfg: DictConfig):
    print("======== Experimental Configuration ========")
    print(OmegaConf.to_yaml(cfg))
    print("============================================")
    
    # load the model and processor with 4-bit quanitzation
    processor = load_processor()
    print(f"Loading Base VLM: {cfg.model.model_id}...")
    base_model = load_vlm_model()
    
    # model's gradients for quantized training
    base_model = prepare_model_for_kbit_training(base_model)
    
    # 2. Inject LoRA Adapters
    peft_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        target_modules=list(cfg.model.target_modules),
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # initialize dataset - this will download and prepare the SROIE dataset, including offline resizing if configured
    print("\nInitializing Kaggle SROIE Datasets...")
    raw_data = SROIEDataset(
        manifest=list(cfg.data.manifest),
        resize_offline=cfg.data.resize_offline,
        max_edge=cfg.data.max_edge,
        seed=cfg.data.seed
    )
    
    # this gives the training dataset in the format needed for Trainer
    train_dataset = SROIEMultimodalDataset(
        data_frame=raw_data.train,
        split="train",
        max_edge=cfg.data.max_edge,
        max_seq_length=cfg.model.max_seq_length,
        processor=processor
    )
    
    # configure Training Hyperparameters
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=5,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        fp16=True, 
        remove_unused_columns=False, # CRITICAL: Keeps custom vision tensors from being deleted
        report_to="none"
    )
    
    # trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vlm_collate_fn,
    )

    print("\nPipeline completed. Launching training loop...")
    trainer.train()

if __name__ == "__main__":
    with initialize(version_base=None, config_path="../config"):
        # load and compose the config file as the mentioned in config.yaml file
        cfg = compose(config_name="config")

    train_model(cfg)