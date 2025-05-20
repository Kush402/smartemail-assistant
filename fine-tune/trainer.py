# fine-tune/trainer.py

import os
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from dataset_loader import load_dataset_for_training

# Load environment/config
MODEL_NAME = "gpt2"
OUTPUT_DIR = "model/checkpoints"
ADAPTER_DIR = "../model/peft_adapter"
PROCESSED_DATA_PATH = "../data/processed/processed_emails.csv"

def main():
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    # Step 1: Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = prepare_model_for_kbit_training(model)

    # Step 2: Apply LoRA with PEFT
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Step 3: Load dataset
    print("Loading dataset...")
    train_dataset, _ = load_dataset_for_training(PROCESSED_DATA_PATH, tokenizer_name=MODEL_NAME)

    # Step 4: Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
    )

    # Step 5: Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Step 6: Start training
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Step 7: Save model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save LoRA adapter
    print("Saving LoRA adapter...")
    model.save_pretrained(ADAPTER_DIR)
    
    print(f"✅ Model saved to {OUTPUT_DIR}")
    print(f"✅ LoRA adapter saved to {ADAPTER_DIR}")

if __name__ == "__main__":
    main()
