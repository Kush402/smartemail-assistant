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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset_loader import load_dataset_for_training

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_NAME = config['model']['name']
OUTPUT_DIR = config['training']['output_dir']
ADAPTER_DIR = "model/peft_adapter"
PROCESSED_DATA_PATH = "data/processed/processed_emails.csv"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=config['model']['use_4bit']
    )
    model = prepare_model_for_kbit_training(model)

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
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],
        fp16=config['training']['fp16'],
        report_to="none",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
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
