# TinyStories: Scratch vs Fine-Tuned Training Comparison

Compares training a language model **from random weights** against **fine-tuning pretrained weights** using the same TinyStories-1M architecture on an entity extraction task (name, date, payment amount from natural language).

---

## Objective

Given a sentence like:
```
John Smith visited on March 5 and paid $250.
```
Output structured JSON:
```json
{"name": "John Smith", "date": "March 5", "money_amount": "$250"}
```

---

## Repository Structure

```
├── download_model.py                  # Download TinyStories-1M from Hugging Face
├── tinystories_scratch_training.py    # Train from random weights
├── tinystories_pretrained_training.py # Fine-tune pretrained weights
├── data.json                          # Training data (361 unique examples)
├── TinyStories-1M-local/              # Downloaded model (created by download_model.py)
├── tinystories_finetuned/             # Output: scratch-trained model
└── tinystories_finetuned_pretrained/  # Output: fine-tuned model
```

---

## Setup

```bash
pip install torch transformers
python download_model.py
```

---

## Usage

```bash
# Train from scratch (~75 epochs to converge)
python tinystories_scratch_training.py --mode both

# Fine-tune pretrained model (~43 epochs to converge)
python tinystories_pretrained_training.py --mode both

# Inference only (after training)
python tinystories_scratch_training.py --mode infer
python tinystories_pretrained_training.py --mode infer
```

**CLI flags:** `--mode [train|infer|both]` · `--data_file data.json` · `--save_dir ./output_dir`

---

## Results

### Convergence Comparison

| | Scratch | Pretrained |
|---|---|---|
| Starting loss | ~9.3 | ~5.0 |
| Epochs to reach loss < 0.08 | **75** | **43** |
| Total training steps | ~3,450 | ~1,978 |
| Speedup | — | **~1.7×** |

### Inference Quality (at convergence)

**Scratch model** — partial generalization; gets date/amount right, name drifts on unseen inputs:
```
Input : John Sarah visited on March 5 and paid $297.
Output: {"name": "John Smith", "date": "March 5", "money_amount": "$297"}  ✓ date/amount, ✗ name

Input : Patrick Lee checked in on July 4 and paid $189.99.
Output: {"name": "Riley Lee", "date": "July 4", "money_amount": "$99.99"}  ✗ name, ✗ amount
```

**Pretrained model** — accurate on seen examples, partial generalization on unseen:
```
Input : John Smith visited on March 5 and paid $250.
Output: {"name": "John Smith", "date": "March 5", "money_amount": "$250"}  ✓ perfect

Input : Carlos Mendoza stopped by on February 22 and paid $340.50.
Output: {"name": "Bella Mendoza", "date": "February 22", "money_amount": "$340.50"}  ✓ date/amount, ✗ first name
```

---

## Key Observations

**Pretrained weights converge faster** because the model already understands language structure — it only needs to learn the extraction pattern, not language from scratch.

**Both models learn the output format** (valid JSON with correct keys) well before they learn to copy exact values from the input. Format learning (~loss 2.0) precedes value extraction (~loss < 0.1).

**True generalization to unseen names is limited** for a 3.7M parameter model. Both models partially interpolate from training data rather than learning a pure copy mechanism. A larger model or explicit copy/pointer mechanism would be needed for robust out-of-distribution performance.

---

## Architecture

- **Model:** `roneneldan/TinyStories-1M` — 3.75M parameters, GPT-Neo style transformer
- **Task format:** Causal LM with prompt masking (loss computed on output tokens only)
- **Optimizer:** AdamW with Cosine Annealing Warm Restarts
- **Prompt template:**
  ```
  Input: <sentence>
  Output:
  {"name": "...", "date": "...", "money_amount": "..."}
  ```

---

## Model Weights

Saved models are not committed to this repo. Run `download_model.py` then the training script to reproduce them.
