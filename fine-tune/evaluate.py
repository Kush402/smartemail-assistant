import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer."""
    print("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_model, model_path)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=200):
    """Generate a response for the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.3,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_single_response(model, tokenizer, subject, email_body):
    """Generate a response for a single email."""
    prompt = f"""You are an HR manager. Write a response to this leave request email. Follow this exact format:

1. Start with "Dear [Name],"
2. Acknowledge the leave request
3. Confirm the dates mentioned
4. State approval or request more information
5. End with a professional closing

Example format:
Dear [Name],

I have received your leave request for [dates]. [Approval/Request for more info].

[Additional details if needed]

Best regards,
[Your name]

Now, respond to this email:

Subject: {subject}

Original Email:
{email_body}

Response:"""
    response = generate_response(model, tokenizer, prompt)
    return response

def evaluate_model(model_path, test_data_path):
    """Evaluate the model on test data."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    test_dataset = load_dataset("csv", data_files=test_data_path)["train"]
    
    # Evaluate
    print("Evaluating model...")
    results = []
    
    for example in tqdm(test_dataset):
        prompt = f"Write a professional email response to:\nSubject: {example['subject']}\n\nOriginal Email:\n{example['body']}\n\nResponse:"
        generated_response = generate_response(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "generated": generated_response,
            "reference": example["body"]
        })
    
    return results

if __name__ == "__main__":
    MODEL_PATH = "model/checkpoints"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    model.eval()
    
    # Interactive input
    print("\nSmartEmail Assistant - Professional Email Response Generator\n")
    print("This assistant will help you respond to leave request emails professionally.")
    print("Please enter the details below:\n")
    
    subject = input("Enter the email subject: ")
    print("\nEnter the original email body (end with an empty line):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    email_body = "\n".join(lines)
    
    print("\nGenerating professional response...\n")
    response = generate_single_response(model, tokenizer, subject, email_body)
    
    print("Generated Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)

# Model evaluation metrics 