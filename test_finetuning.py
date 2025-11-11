#!/usr/bin/env python3
"""
Unit Test for LoRA Fine-Tuning

This script tests the fine-tuning pipeline on a very small subset of data
with just a few iterations to validate that the training process works correctly.
"""

import torch
import numpy as np
from datasets import load_dataset
from lora_finetuning import (
    setup_lora_model,
    prepare_dataset,
    train_lora_model,
    evaluate_model
)


def test_finetuning_pipeline():
    """
    Test the complete fine-tuning pipeline with minimal data and iterations.
    """
    print("=" * 80)
    print("LoRA Fine-Tuning Unit Test")
    print("=" * 80)
    print("\nThis test uses:")
    print("  - 10 training samples")
    print("  - 5 validation samples")
    print("  - 2 training epochs")
    print("  - Batch size of 2")
    print("=" * 80)
    print()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
    OUTPUT_DIR = "./test_lora_output"
    NUM_TRAIN_SAMPLES = 10
    NUM_VAL_SAMPLES = 5
    NUM_EPOCHS = 2
    BATCH_SIZE = 2
    MAX_LENGTH = 128  # Shorter sequences for testing

    print("\n[1/6] Loading model and setting up LoRA...")
    print("-" * 80)
    model, tokenizer, lora_config = setup_lora_model(
        model_name=MODEL_NAME,
        lora_r=8,  # Smaller rank for testing
        lora_alpha=16,
        lora_dropout=0.05
    )

    print("\n[2/6] Loading and preparing dataset...")
    print("-" * 80)
    # Load full dataset first
    tokenized_train_full, tokenized_val_full, raw_dataset = prepare_dataset(
        dataset_name='shawhin/imdb-truncated',
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    # Create small subsets for testing
    tokenized_train = tokenized_train_full.select(range(NUM_TRAIN_SAMPLES))
    tokenized_val = tokenized_val_full.select(range(NUM_VAL_SAMPLES))

    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(tokenized_train)}")
    print(f"  Validation samples: {len(tokenized_val)}")

    print("\n[3/6] Evaluating base model (before fine-tuning)...")
    print("-" * 80)
    base_acc, base_correct, base_total = evaluate_model(
        model,
        tokenizer,
        raw_dataset,
        num_samples=NUM_VAL_SAMPLES,
        debug=False
    )

    print(f"\nBase Model Results:")
    print(f"  Accuracy: {base_acc:.2%} ({base_correct}/{base_total})")

    print("\n[4/6] Training model with LoRA...")
    print("-" * 80)
    trainer = train_lora_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=2e-4,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        logging_steps=2
    )

    print("\n[5/6] Evaluating fine-tuned model...")
    print("-" * 80)
    ft_acc, ft_correct, ft_total = evaluate_model(
        model,
        tokenizer,
        raw_dataset,
        num_samples=NUM_VAL_SAMPLES,
        debug=False
    )

    print(f"\nFine-Tuned Model Results:")
    print(f"  Accuracy: {ft_acc:.2%} ({ft_correct}/{ft_total})")

    print("\n[6/6] Test Results Summary")
    print("=" * 80)
    print(f"\nBase Model Accuracy:       {base_acc:.2%} ({base_correct}/{base_total})")
    print(f"Fine-Tuned Model Accuracy: {ft_acc:.2%} ({ft_correct}/{ft_total})")

    if ft_acc > base_acc:
        improvement = ft_acc - base_acc
        print(f"\n✓ PASS: Model improved by {improvement:.2%}")
    elif ft_acc == base_acc:
        print(f"\n⚠ WARNING: No improvement detected (may need more training)")
    else:
        print(f"\n✗ FAIL: Model performance decreased")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

    # Clean up
    print("\nNote: Model checkpoints saved to:", OUTPUT_DIR)
    print("You can delete this directory to free up space.")

    return {
        'base_accuracy': base_acc,
        'finetuned_accuracy': ft_acc,
        'improvement': ft_acc - base_acc,
        'test_passed': ft_acc >= base_acc
    }


if __name__ == "__main__":
    try:
        results = test_finetuning_pipeline()
        exit_code = 0 if results['test_passed'] else 1
        exit(exit_code)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"ERROR: Test failed with exception:")
        print(f"{type(e).__name__}: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        exit(1)
