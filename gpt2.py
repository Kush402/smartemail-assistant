from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "gpt2"  # base model used for fine-tuning

# Save the base model and tokenizer for compatibility with your app
AutoModelForCausalLM.from_pretrained(model_id).save_pretrained("model/checkpoints/gpt2")
AutoTokenizer.from_pretrained(model_id).save_pretrained("model/tokenizer")
