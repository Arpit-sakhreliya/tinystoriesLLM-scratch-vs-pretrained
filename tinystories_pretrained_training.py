"""
tinystories_pretrained_training.py
-------------------
Same as train_and_infer.py but loads PRETRAINED weights from TinyStories-1M
instead of random weights. Converges much faster since the model already
understands language structure.
"""

import json
import os
import re
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
MODEL_NAME  = "./TinyStories-1M-local"
DATA_FILE   = "data.json"
SAVE_DIR    = "./tinystories_finetuned_pretrained"
EPOCHS      = 50             # far fewer needed — pretrained weights converge fast
BATCH_SIZE  = 8
LR          = 2e-4           # lower LR than random-init — don't destroy pretrained weights
MAX_SEQ_LEN = 128
LOG_EVERY   = 50
LOSS_TARGET = 0.08


def resolve_local(path):
    return os.path.abspath(path) if os.path.exists(path) else path


SEP = "\nOutput:\n"

def make_prompt(text):
    return f"Input: {text.strip()}{SEP}"

def make_full(text, out_dict):
    return make_prompt(text) + json.dumps(out_dict, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
# DATASET  (identical to train_and_infer.py)
# ──────────────────────────────────────────────────────────────
class ExtractionDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len):
        self.examples = []
        with open(data_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        try:
            records = json.loads(raw)
            if isinstance(records, dict):
                records = [records]
        except json.JSONDecodeError:
            records = [json.loads(l) for l in raw.splitlines() if l.strip()]

        seen, unique = set(), []
        for r in records:
            if r["input"] not in seen:
                seen.add(r["input"])
                unique.append(r)
        print(f"  Unique examples: {len(unique)}")

        pad_id = tokenizer.pad_token_id
        for rec in unique:
            full_ids   = tokenizer.encode(make_full(rec["input"], rec["output"]),
                                          add_special_tokens=True)
            prompt_ids = tokenizer.encode(make_prompt(rec["input"]),
                                          add_special_tokens=False)
            n_prompt   = len(prompt_ids)
            full_ids   = full_ids[:max_len]
            labels     = [-100] * n_prompt + full_ids[n_prompt:]
            labels     = labels[:max_len]
            pad_len    = max_len - len(full_ids)
            self.examples.append({
                "input_ids":      torch.tensor(full_ids + [pad_id]*pad_len,   dtype=torch.long),
                "attention_mask": torch.tensor([1]*len(full_ids)+[0]*pad_len, dtype=torch.long),
                "labels":         torch.tensor(labels   + [-100]*pad_len,     dtype=torch.long),
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


# ──────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = resolve_local(MODEL_NAME)

    # ── tokenizer ──
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # ── KEY DIFFERENCE: load pretrained weights, not random ──
    print("Loading PRETRAINED model weights …")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))  # accommodate new <pad> token
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── data ──
    print(f"Loading dataset from {args.data_file} …")
    dataset    = ExtractionDataset(args.data_file, tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"  {len(dataloader)} batches/epoch")

    # ── optimizer: cosine restarts ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    model.train()
    global_step = 0
    print("\n===== TRAINING =====")
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask,
                         labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1
            if global_step % LOG_EVERY == 0:
                print(f"  step {global_step:5d}  loss={loss.item():.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        avg = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:3d}/{EPOCHS}  avg_loss={avg:.4f}")

        if avg <= LOSS_TARGET:
            print(f"  >> Reached loss target {LOSS_TARGET} — stopping early.")
            break

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to {save_dir}")


# ──────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────
def infer(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = resolve_local(args.save_dir)
    print(f"Loading model from {save_dir} …")

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model     = AutoModelForCausalLM.from_pretrained(save_dir)
    model.to(device).eval()

    test_inputs = [
        "John Smith visited on March 5 and paid $250.",
        "Sarah Johnson arrived on April 12 and paid $125.50.",
        "Michael Brown visited on January 8 and paid $75.",
        "Patricia Lee checked in on July 4 and paid $199.99.",   # unseen
        "Carlos Mendoza stopped by on February 22 and paid $340.50.",  # unseen
    ]

    print("\n===== INFERENCE =====")
    for text in test_inputs:
        prompt = make_prompt(text)
        enc    = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out_ids = model.generate(
                enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        full      = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        after_sep = full.rsplit("Output:", 1)[-1].strip()
        match     = re.search(r'\{[^{}]*\}', after_sep, re.DOTALL)
        result    = match.group(0) if match else after_sep[:200]
        print(f"\nInput : {text}")
        print(f"Output: {result}")


# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",      choices=["train", "infer", "both"], default="both")
    p.add_argument("--data_file", default=DATA_FILE)
    p.add_argument("--save_dir",  default=SAVE_DIR)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode in ("train", "both"): train(args)
    if args.mode in ("infer", "both"): infer(args)