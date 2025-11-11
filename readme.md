# LoRA Fine-Tuning with SmolLM2-360M

This project demonstrates fine-tuning of a language model using LoRA (Low-Rank Adaptation) for sentiment analysis on movie reviews.

## Overview

- **Base Model:** [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- **Dataset:** [shawhin/imdb-truncated](https://huggingface.co/datasets/shawhin/imdb-truncated/viewer/default/validation?views%5B%5D=validation) (1000 train, 1000 validation samples)
- **Task:** Binary sentiment classification (positive/negative movie reviews)
- **Method:** LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning

## Dataset

The project uses the IMDB truncated dataset for sentiment analysis:

🔗 **[View Dataset on Hugging Face](https://huggingface.co/datasets/shawhin/imdb-truncated/viewer/default/validation?views%5B%5D=validation)**

- **Training samples:** 1,000 movie reviews
- **Validation samples:** 1,000 movie reviews
- **Labels:** Binary (0 = negative, 1 = positive)
- **Format:** Text reviews with sentiment labels

## How to Run the Code

### Option 1: Docker

**Quick Start:**
```bash
# Run with Docker Compose
docker-compose up --build
```

**Or use Docker commands directly:**
```bash
# Build the image
docker build -t lora-test .

# Run with GPU
docker run --gpus all \
  -v $(pwd)/cache:/app/.cache \
  -v $(pwd)/test_lora_output:/app/test_lora_output \
  lora-test

# Run without GPU (CPU only)
docker run \
  -v $(pwd)/cache:/app/.cache \
  -v $(pwd)/test_lora_output:/app/test_lora_output \
  lora-test
```

**Expected Runtime:**
- GPU: 5-10 minutes
- CPU: 30-60 minutes

### Option 2: Local Installation

**Step 1: Install Dependencies**
```bash
pip install -q torch transformers datasets peft accelerate bitsandbytes trl
```

**Step 2: Run the Unit Test**
```bash
python test_finetuning.py
```

This will:
- Load 10 training samples and 5 validation samples
- Fine-tune the model for 2 epochs
- Compare base vs fine-tuned performance
- Exit with code 0 if successful


## Outputs

### Main Training Output: `LoRA_fine_tuning.ipynb`

### Test Results: `test_results.ipynb`


## Module Functions

The `lora_finetuning.py` module provides reusable functions:

### Core Functions

- **`setup_lora_model()`** - Load base model and apply LoRA configuration
- **`prepare_dataset()`** - Load and tokenize dataset with instruction prompts
- **`train_lora_model()`** - Execute training with specified parameters
- **`evaluate_model()`** - Evaluate model on sentiment classification
- **`load_finetuned_model()`** - Load a previously fine-tuned model

### Helper Functions

- **`generate_response()`** - Generate text from the model
- **`create_prompt()`** - Format examples for sentiment analysis
- **`tokenize_function()`** - Tokenize text data
