import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


def load_model_and_tokenizer(model_path, tokenizer_path=None, base_model="gpt2"):
    """Load the fine-tuned model and tokenizer."""
    print("🔁 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=256):
    """Generate a response for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_model(model_path, data_path, tokenizer_path=None, base_model="gpt2"):
    """Evaluate the model using a CSV with prompts and reference completions."""
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, base_model)

    print(f"📂 Loading test data from: {data_path}")
    df = pd.read_csv(data_path)
    if "prompt" not in df.columns:
        raise ValueError("CSV must contain 'prompt' column.")
    
    results = []
    print("🔍 Generating responses...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]
        reference = row.get("completion", "")
        generated = generate_response(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "generated": generated,
            "reference": reference
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("✅ Evaluation complete. Results saved to evaluation_results.csv")
    return results_df


if __name__ == "__main__":
    MODEL_PATH = "model/peft_adapter"
    TEST_DATA_PATH = "data/processed/processed_emails.csv"
    TOKENIZER_PATH = "model/checkpoints"  # Optional

    evaluate_model(MODEL_PATH, TEST_DATA_PATH, TOKENIZER_PATH)
