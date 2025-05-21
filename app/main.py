import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "checkpoints")
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "model", "peft_adapter")

def load_model_and_tokenizer():
    """Load the model and tokenizer with proper configuration."""
    print("Loading model and tokenizer...")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Adapter directory: {ADAPTER_DIR}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, subject, email_body):
    """Generate a professional email response."""
    # Create a more structured prompt
    prompt = f"""You are an HR manager. Write a professional response to the following leave request:

Subject: {subject}

Body:
{email_body}

Response (start with "Dear [Name],"):"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response with better parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and clean up response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    
    return response

def main():
    """Main function to run the email assistant."""
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        print("\nSmartEmail Assistant - Professional Email Response Generator")
        print("Type 'exit' to quit\n")
        
        while True:
            # Get email subject
            subject = input("Enter email subject: ").strip()
            if subject.lower() == 'exit':
                break
            
            # Get email body
            print("\nEnter email body (end with an empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            email_body = "\n".join(lines)
            
            if not email_body.strip():
                print("Email body cannot be empty. Please try again.")
                continue
            
            # Generate and display response
            print("\nGenerated Response:")
            print("-" * 80)
            response = generate_response(model, tokenizer, subject, email_body)
            print(response)
            print("-" * 80)
            print()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nThank you for using SmartEmail Assistant!")

if __name__ == "__main__":
    main()
