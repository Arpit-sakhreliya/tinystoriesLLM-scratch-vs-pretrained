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

"""
infer.py
---------
Run inference with a locally saved TinyStories-1M model.
"""

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_PATH = r"D:\python\tets\LLMs\tinystoriesLLM-scratch-vs-pretrained\TinyStories-1M-local"

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# # Use GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# prompt = "Once upon a time, there was a little dragon"

# # Tokenize
# inputs = tokenizer(prompt, return_tensors="pt").to(device)

# # Generate
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=100,
#         temperature=0.8,
#         top_p=0.95,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,
#     )

# # Decode
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print("\nGenerated text:\n")
# print(text)




'''
Architecture of the TinyStories-1M Model
--------------------------------------
This file provides an overview of the architecture of the TinyStories-1M model, including its components and parameters.
'''

# from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# MODEL_NAME = r"D:\python\tets\LLMs\tinystoriesLLM-scratch-vs-pretrained\TinyStories-1M-local"

# # ─────────────────────────────
# # 1. Load tokenizer
# # ─────────────────────────────
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# vocab_size = len(tokenizer)

# print("🔹 TOKENIZER INFO")
# print("Vocabulary size:", vocab_size)
# print("Special tokens:", tokenizer.all_special_tokens)
# print("Max length:", tokenizer.model_max_length)


# # ─────────────────────────────
# # 2. Load model config (NO weights needed)
# # ─────────────────────────────
# config = AutoConfig.from_pretrained(MODEL_NAME)

# print("\n🔹 MODEL CONFIG INFO")
# print("Embedding dimension (hidden_size):", config.hidden_size)


# # ─────────────────────────────
# # 3. Load model (for embedding + params)
# # ─────────────────────────────
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# embedding_matrix = model.get_input_embeddings().weight

# print("\n🔹 EMBEDDING MATRIX INFO")
# print("Shape:", embedding_matrix.shape)
# print("Meaning: (vocab_size, embedding_dim)")
# print("Embedding dimension:", embedding_matrix.shape[1])


# # ─────────────────────────────
# # 4. TOTAL PARAMETERS
# # ─────────────────────────────
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print("\n🔹 MODEL PARAMETERS")
# print("Total parameters     :", f"{total_params:,}")
# print("Trainable parameters :", f"{trainable_params:,}")


# # ─────────────────────────────
# # 5. (Optional) breakdown per module
# # ─────────────────────────────
# print("\n🔹 PARAMETER BREAKDOWN (first 20 layers)")
# for i, (name, param) in enumerate(model.named_parameters()):
#     print(f"{name:60s} {param.numel():,}")
#     if i >= 10:
#         print(" there are more parameters...")
#         break

# print("\n🔹 PARAMETER BREAKDOWN (ALL LAYERS)")

# for name, param in model.named_parameters():
#     print(f"{name:60s} {param.numel():,}")

# print("🔹 TOTAL LAYERS:", config.num_hidden_layers)
# ─────────────────────────────
# 6. SIMPLE TOKENIZER TEST (ADDED)
# ─────────────────────────────
# This lets you type text and see how tokenizer converts it into tokens and IDs
# It helps understand what the model actually "sees"

# def test_tokenizer():
#     print("\n🔹 TOKENIZER TEST MODE (type 'exit' to stop)")

#     while True:
#         text = input("\nEnter text: ")

#         if text.lower() == "exit":
#             break

#         # Convert text → tokens (words/subwords)
#         tokens = tokenizer.tokenize(text)

#         # Convert text → token IDs (numbers used by model)
#         ids = tokenizer.encode(text)

#         # Decode back to check reversibility
#         decoded = tokenizer.decode(ids)

#         print("\nTokens   :", tokens)
#         print("Token IDs:", ids)
#         print("Decoded  :", decoded)
#         print("-" * 50)


# # Run interactive tokenizer test
# test_tokenizer()