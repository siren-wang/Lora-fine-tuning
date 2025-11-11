# LoRA Fine-Tuning Module

This directory contains a modular implementation of LoRA (Low-Rank Adaptation) fine-tuning for language models.

## Files

- **`lora_finetuning.py`** - Main module with reusable functions for LoRA fine-tuning
- **`test_finetuning.py`** - Unit test script that validates the training pipeline with minimal data
- **`example_usage.py`** - Example script showing how to use the module
- **`LoRA-fine-tuning.ipynb`** - Original Jupyter notebook (kept for reference)

## Module Functions

### Core Functions

- `setup_lora_model()` - Load a base model and apply LoRA configuration
- `prepare_dataset()` - Load and prepare dataset for fine-tuning
- `train_lora_model()` - Train a model with LoRA
- `evaluate_model()` - Evaluate model accuracy on sentiment classification
- `load_finetuned_model()` - Load a previously fine-tuned model

### Helper Functions

- `generate_response()` - Generate text from a model
- `create_prompt()` - Format examples for sentiment analysis
- `tokenize_function()` - Tokenize text data

## Quick Start

### Running the Unit Test

Test the fine-tuning pipeline with minimal data (10 train samples, 5 validation samples, 2 epochs):

```bash
python test_finetuning.py
```

This will:
- Load the base model
- Train on a tiny subset
- Compare base vs fine-tuned performance
- Exit with code 0 if test passes

### Using the Module

```python
from lora_finetuning import (
    setup_lora_model,
    prepare_dataset,
    train_lora_model,
    evaluate_model
)

# Setup model with LoRA
model, tokenizer, _ = setup_lora_model(
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
    lora_r=16,
    lora_alpha=32
)

# Prepare data
train_data, val_data, dataset = prepare_dataset(
    dataset_name='shawhin/imdb-truncated',
    tokenizer=tokenizer
)

# Train
trainer = train_lora_model(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    output_dir="./output",
    num_epochs=3
)

# Evaluate
accuracy, correct, total = evaluate_model(
    model, tokenizer, dataset, num_samples=100
)
print(f"Accuracy: {accuracy:.2%}")
```

### Full Example

See `example_usage.py` for a complete working example:

```bash
python example_usage.py
```

## Requirements

```bash
pip install transformers datasets peft accelerate bitsandbytes trl torch
```

## Configuration

### LoRA Parameters

- `lora_r`: Rank of low-rank matrices (default: 16, test: 8)
- `lora_alpha`: Scaling factor (default: 32, test: 16)
- `lora_dropout`: Dropout rate (default: 0.05)
- `target_modules`: Modules to adapt (default: ["q_proj", "v_proj", "k_proj", "o_proj"])

### Training Parameters

- `num_epochs`: Number of training epochs (default: 3, test: 2)
- `batch_size`: Per-device batch size (default: 4, test: 2)
- `learning_rate`: Learning rate (default: 2e-4)
- `max_length`: Maximum sequence length (default: 256, test: 128)

## Testing

The unit test (`test_finetuning.py`) uses:
- **10 training samples** (vs 1000 in full training)
- **5 validation samples** (vs 1000 in full training)
- **2 epochs** (vs 3 in full training)
- **Smaller LoRA rank** (8 vs 16)
- **Shorter sequences** (128 vs 256 tokens)

This allows quick validation of the pipeline without requiring significant compute resources or time.

## Performance

Expected results on full dataset (1000 train/val samples, 3 epochs):
- Base model accuracy: ~66%
- Fine-tuned model accuracy: ~82%
- Improvement: ~16% absolute, ~24% relative

## License

Same as the parent project.
