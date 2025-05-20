import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model paths
MODEL_DIR = "../model/checkpoints"
ADAPTER_DIR = "../model/peft_adapter"

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    print("Loading model and tokenizer...")
    
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=200):
    """Generate a response for the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    print("\nSmartEmail Assistant - Professional Email Response Generator")
    print("Type 'exit' to quit\n")
    
    while True:
        # Get user input
        subject = input("Enter email subject: ")
        if subject.lower() == 'exit':
            break
            
        print("\nEnter email body (end with an empty line):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        email_body = "\n".join(lines)
        
        # Generate response
        prompt = f"Write a professional email response to:\nSubject: {subject}\n\nOriginal Email:\n{email_body}\n\nResponse:"
        response = generate_response(model, tokenizer, prompt)
        
        print("\nGenerated Response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        print()

if __name__ == "__main__":
    main()
