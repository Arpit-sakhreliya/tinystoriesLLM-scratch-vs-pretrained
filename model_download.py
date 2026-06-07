'''
download_model.py
-------------------
Downloads the TinyStories-1M model from Hugging Face and saves it locally.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "roneneldan/TinyStories-1M"
save_path = "./TinyStories-1M-local"

print("Downloading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Saving locally...")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"Done! Saved to {save_path}")