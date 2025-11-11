"""
LoRA Fine-Tuning Utilities

This module contains utility functions for fine-tuning language models using LoRA
(Low-Rank Adaptation) for sentiment analysis tasks.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 0.1
) -> str:
    """
    Generate response from model given a prompt.

    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for the model
        prompt: The input prompt text
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation

    Returns:
        The generated response text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def evaluate_model(
    model,
    tokenizer,
    dataset,
    num_samples: int = 100,
    debug: bool = False
) -> Tuple[float, int, int]:
    """
    Evaluate model accuracy on sentiment classification.

    Args:
        model: The language model to evaluate
        tokenizer: The tokenizer for the model
        dataset: The dataset to evaluate on (must have 'validation' split)
        num_samples: Number of samples to evaluate
        debug: Whether to print debug information

    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    correct = 0
    total = 0

    samples_to_test = min(num_samples, len(dataset['validation']))

    print(f"Evaluating on {samples_to_test} validation samples...\n")

    for i in range(samples_to_test):
        sample = dataset['validation'][i]
        review_text = sample['text']
        true_label = sample['label']
        true_sentiment = "positive" if true_label == 1 else "negative"

        prompt = """You are a sentiment classifier.
Read each review and respond only with 'positive' or 'negative'.

Example 1:
Review: This movie was boring and too long.
Answer: negative

Example 2:
Review: I loved the characters and story.
Answer: positive

Now classify this one:
Review: """ + review_text + "\nAnswer:"

        # Generate prediction
        output = generate_response(model, tokenizer, prompt, max_new_tokens=10)

        # Extract only the generated part (after the prompt)
        generated_text = output[len(prompt):].strip()

        # Extract predicted sentiment
        generated_lower = generated_text.lower()
        if "positive" in generated_lower and "negative" not in generated_lower:
            predicted_sentiment = "positive"
        elif "negative" in generated_lower and "positive" not in generated_lower:
            predicted_sentiment = "negative"
        else:
            predicted_sentiment = None

        if predicted_sentiment == true_sentiment:
            correct += 1

        # Show first 5 examples with debug info
        if i < 5:
            print(f"Example {i+1}:")
            print(f"  Review: {review_text[:100]}...")
            print(f"  True sentiment: {true_sentiment}")
            if debug:
                print(f"  Raw output: {generated_text}")
            print(f"  Predicted: {predicted_sentiment}")
            print(f"  Correct: {'✓' if predicted_sentiment == true_sentiment else '✗'}")
            print()

        total += 1

        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{samples_to_test} samples processed...")

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def create_prompt(example: Dict) -> Dict:
    """
    Create instruction-formatted prompt for sentiment analysis.

    Args:
        example: Dataset example with 'text' and 'label' fields

    Returns:
        Dictionary with formatted 'text' field
    """
    sentiment = "positive" if example['label'] == 1 else "negative"

    # Instruction format
    prompt = """You are a sentiment classifier.
Read each review and respond only with 'positive' or 'negative'.

Example 1:
Review: This movie was boring and too long.
Answer: negative

Example 2:
Review: I loved the characters and story.
Answer: positive

Now classify this one:
Review: """ + example['text'] + "\nAnswer:"

    return {"text": prompt}


def tokenize_function(examples: Dict, tokenizer, max_length: int = 256) -> Dict:
    """
    Tokenize the text data.

    Args:
        examples: Batch of examples with 'text' field
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs and labels
    """
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_dataset(
    dataset_name: str = 'shawhin/imdb-truncated',
    tokenizer = None,
    max_length: int = 256
):
    """
    Load and prepare dataset for fine-tuning.

    Args:
        dataset_name: Name of the dataset to load
        tokenizer: Tokenizer to use for preprocessing
        max_length: Maximum sequence length

    Returns:
        Tuple of (tokenized_train, tokenized_val, raw_dataset)
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Format datasets
    formatted_train = dataset['train'].map(create_prompt, remove_columns=['label'])
    formatted_val = dataset['validation'].map(create_prompt, remove_columns=['label'])

    # Tokenize datasets
    tokenized_train = formatted_train.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=['text']
    )

    tokenized_val = formatted_val.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=['text']
    )

    return tokenized_train, tokenized_val, dataset


def setup_lora_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules = None
):
    """
    Load base model and apply LoRA configuration.

    Args:
        model_name: Name or path of the base model
        lora_r: Rank of the low-rank matrices
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability
        target_modules: List of module names to apply LoRA to

    Returns:
        Tuple of (model, tokenizer, lora_config)
    """
    # Default target modules for transformer models
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    print("LoRA Model Configuration:")
    print(lora_config)
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()

    return model, tokenizer, lora_config


def train_lora_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str = "./lora_finetuned",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    logging_steps: int = 50
) -> Trainer:
    """
    Train a LoRA model with the specified parameters.

    Args:
        model: The PEFT model to train
        tokenizer: The tokenizer for the model
        train_dataset: Training dataset (tokenized)
        eval_dataset: Evaluation dataset (tokenized)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        gradient_accumulation_steps: Steps to accumulate gradients
        warmup_steps: Number of warmup steps
        logging_steps: Steps between logging

    Returns:
        The trained Trainer object
    """
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        push_to_hub=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Train the model
    trainer.train()

    print(f"\nTraining completed!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return trainer


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str
):
    """
    Load a fine-tuned LoRA model.

    Args:
        base_model_name: Name or path of the base model
        adapter_path: Path to the LoRA adapter weights

    Returns:
        The loaded PEFT model
    """
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load fine-tuned model (base + LoRA adapter)
    finetuned_model = PeftModel.from_pretrained(
        base_model,
        adapter_path
    )

    return finetuned_model
