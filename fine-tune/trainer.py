# fine-tune/trainer.py

import os
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset_loader import load_dataset_for_training

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configuration
config_path = os.path.join(SCRIPT_DIR, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

MODEL_NAME = config['model']['name']
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), config['training']['output_dir'])
ADAPTER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "model/peft_adapter")

# Handle both local and Colab environments
if os.path.exists('/content'):  # Running in Colab
    PROCESSED_DATA_PATH = '/content/smartemail-assistant/data/processed/processed_emails.csv'
else:  # Running locally
    PROCESSED_DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "data/processed/processed_emails.csv")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure 4-bit quantization
    if config['model']['use_4bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None

    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if config['training']['fp16'] else torch.float32,
        device_map="auto",  # Let the model decide device placement
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    # Apply LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )
    model = get_peft_model(model, lora_config)

    # Load training data
    print("Loading dataset...")
    print(f"Looking for data at: {PROCESSED_DATA_PATH}")
    train_dataset, _ = load_dataset_for_training(PROCESSED_DATA_PATH, tokenizer_name=MODEL_NAME)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        fp16=config['training']['fp16'],
        report_to="none",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        # Add these for better GPU utilization
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # Save
    print("Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(ADAPTER_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")
    print(f"✅ LoRA adapter saved to {ADAPTER_DIR}")

if __name__ == "__main__":
    main()
