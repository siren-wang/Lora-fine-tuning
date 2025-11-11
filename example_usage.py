#!/usr/bin/env python3
"""
Example Usage of LoRA Fine-Tuning Module

This script demonstrates how to use the lora_finetuning module
to fine-tune a language model using LoRA.
"""

import torch
import numpy as np
from lora_finetuning import (
    setup_lora_model,
    prepare_dataset,
    train_lora_model,
    evaluate_model,
    load_finetuned_model
)


def main():
    """Main fine-tuning workflow."""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
    OUTPUT_DIR = "./lora_finetuned_smollm2"

    # Step 1: Setup LoRA model
    print("Setting up model with LoRA...")
    model, tokenizer, lora_config = setup_lora_model(
        model_name=MODEL_NAME,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )

    # Step 2: Prepare dataset
    print("\nPreparing dataset...")
    tokenized_train, tokenized_val, raw_dataset = prepare_dataset(
        dataset_name='shawhin/imdb-truncated',
        tokenizer=tokenizer,
        max_length=256
    )

    print(f"Training samples: {len(tokenized_train)}")
    print(f"Validation samples: {len(tokenized_val)}")

    # Step 3: Evaluate base model (optional)
    print("\nEvaluating base model...")
    base_acc, base_correct, base_total = evaluate_model(
        model,
        tokenizer,
        raw_dataset,
        num_samples=100
    )
    print(f"Base model accuracy: {base_acc:.2%}")

    # Step 4: Train the model
    print("\nTraining model...")
    trainer = train_lora_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4
    )

    # Step 5: Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    ft_acc, ft_correct, ft_total = evaluate_model(
        model,
        tokenizer,
        raw_dataset,
        num_samples=100
    )
    print(f"Fine-tuned model accuracy: {ft_acc:.2%}")

    # Step 6: Show improvement
    print("\n" + "=" * 80)
    print("RESULTS:")
    print(f"Base Model:       {base_acc:.2%}")
    print(f"Fine-Tuned Model: {ft_acc:.2%}")
    print(f"Improvement:      {(ft_acc - base_acc):.2%}")
    print("=" * 80)

    # Optional: Load the model later
    # finetuned_model = load_finetuned_model(MODEL_NAME, OUTPUT_DIR)


if __name__ == "__main__":
    main()
