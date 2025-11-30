# AIAA3102 Python Programming for AI - Final Project

## 📋 Project Overview

This project is the final project for the AIAA3102 course, focusing on **Customer Service Intent Classification and Response Generation** tasks.  The project uses the Qwen2. 5 large language model with Supervised Fine-Tuning (SFT) techniques to achieve intelligent processing of customer support conversations, and conducts comparative experiments with baseline methods (0-shot, Few-shot). 

## 🗂️ Project Structure

```
.    
├── training/                      # Model training module
│   ├── classification_sft.py      # Intent classification SFT training
│   ├── response_generation_sft.py # Response generation SFT training
│   ├── run_benchmark.py           # Benchmark running script
│   ├── benchmark_utils.py         # Benchmark utility functions
│   └── visualize_results.py       # Results visualization script
│
├── eval/                          # Model evaluation module
│   ├── classification_eval.py     # Intent classification evaluation
│   ├── classification_eval_robust.py # Robust classification evaluation
│   ├── response_generation_eval.py   # Response generation evaluation
│   ├── config.json                # Evaluation configuration file
│   └── prompt. json                # Prompt template configuration
│
├── infer/                         # Fine-tuned model inference module
│   ├── task1_inference.py         # Task 1 (intent classification) inference
│   └── task2_inference.py         # Task 2 (response generation) inference
│
├── comparison_infer/              # Baseline comparison inference module
│   ├── task1_baseline_0shot.py    # Task 1 baseline inference (0-shot)
│   ├── task1_baseline_10shot.py   # Task 1 baseline inference (10-shot)
│   ├── task2_baseline_0shot. py    # Task 2 baseline inference (0-shot)
│   └── task2_baseline_10shot.py   # Task 2 baseline inference (10-shot)
│
├── assets/                        # Dataset files directory
│   ├── train. jsonl                # Training set
│   ├── validation.jsonl           # Validation set
│   └── test. jsonl                 # Test set
│
├── output/                        # Inference prediction results directory
│   ├── task1_predictions.jsonl           # Task 1 SFT model predictions
│   ├── task1_baseline_0shot_predictions.jsonl   # Task 1 0-shot predictions
│   ├── task1_baseline_10shot_predictions.jsonl  # Task 1 10-shot predictions
│   ├── task2_predictions. jsonl           # Task 2 SFT model predictions
│   ├── task2_baseline_0shot_predictions.jsonl   # Task 2 0-shot predictions
│   └── task2_baseline_10shot_predictions.jsonl  # Task 2 10-shot predictions
│
├── checkpoint/                    # Model checkpoint directory (generated locally, not uploaded)
│   ├── task1_classification/      # Task 1 classification model weights
│   └── task2_response_generation/ # Task 2 response generation model weights
│
├── models/                        # Pre-trained base models directory (generated locally, not uploaded)
│   └── Qwen2.5-1.5B-Instruct/     # Qwen2. 5 base model
│
├── eval_output/                   # Evaluation output directory
├── benchmark_output/              # Benchmark output directory
│
├── app.py                         # 🌐 Demo backend API service
├── index.html                     # 🌐 Demo frontend page
│
├── split. py                       # Dataset stratified sampling script
├── tokenize.py                    # Qwen2.5 tokenization preprocessing script
├── test. py                        # Test script
│
├── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv  # Original dataset
├── data. jsonl                     # Processed data
└── data_tokenize.jsonl            # Tokenized data
```

## 📊 Dataset

This project uses the **Bitext Customer Support Training Dataset**, containing approximately 27K customer service conversation samples, covering 11 intent categories:

`FEEDBACK`, `REFUND`, `ACCOUNT`, `ORDER`, `SHIPPING`, `SUBSCRIPTION`, `PAYMENT`, `CANCEL`, `DELIVERY`, `INVOICE`, `CONTACT`

### Dataset Split

The dataset has been split into three parts, stored in the `assets/` directory:

| File | Description |
|------|-------------|
| `train.jsonl` | Training set |
| `validation. jsonl` | Validation set |
| `test.jsonl` | Test set |

### Data Format

Each data entry is in JSON format, containing the following fields:

```json
{
  "flags": "BLM",
  "instruction": "User input text",
  "category": "ORDER",
  "intent": "place_order",
  "response": "Customer service response text..."
}
```

| Field | Description |
|-------|-------------|
| `flags` | Data marker |
| `instruction` | User input/query |
| `category` | Intent category (11 classes) |
| `intent` | Specific intent |
| `response` | Expected customer service response |

### Data Preprocessing

1. **Data Split**: Use `split.py` for stratified sampling to ensure consistent category proportions in training and validation sets
   ```bash
   python split.py --input_file data.jsonl --train_file train.jsonl --val_file val.jsonl --test_ratio 0.1
   ```

2.  **Data Tokenization**: Use `tokenize.py` for Qwen2.5 tokenization preprocessing
   ```bash
   python tokenize.py data. jsonl data_tokenize.jsonl --model Qwen/Qwen2. 5-7B-Instruct
   ```

## 🚀 Quick Start

### Requirements

- Python 3.10+
- PyTorch
- Transformers
- Swift (ModelScope)
- Flask (for Demo service)
- tqdm

### Install Dependencies

```bash
pip install torch transformers tqdm ms-swift flask flask-cors
```
or
```bash
pip install -r requirements.txt
```

### Download Base Model

Download the Qwen2.5 base model and place it in the `models/` directory:

```bash
# Create models directory
mkdir -p models

# Download model using huggingface-cli (example)
huggingface-cli download Qwen/Qwen2. 5-1.5B-Instruct --local-dir models/Qwen2. 5-1.5B-Instruct
```

Or download from ModelScope:
```bash
# Using modelscope
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')
```

> ⚠️ **Note**: The `models/` directory is not uploaded to GitHub due to the large file size. Please download the base model manually before running. 

## 📝 Task Description

### Task 1: Intent Classification

Predict customer service intent categories based on user input (11-class classification task)

#### SFT Fine-tuning Method
```bash
# Training
python training/classification_sft.py

# Inference
python infer/task1_inference. py

# Evaluation
python eval/classification_eval.py
```

#### Baseline Comparison Methods
```bash
# 0-shot (no examples)
python comparison_infer/task1_baseline_0shot. py

# 10-shot (1 example per class, 11 total)
python comparison_infer/task1_baseline_10shot.py
```

### Task 2: Response Generation

Generate appropriate customer service responses based on user questions and categories

#### SFT Fine-tuning Method
```bash
# Training
python training/response_generation_sft.py

# Inference
python infer/task2_inference.py

# Evaluation
python eval/response_generation_eval.py
```

#### Baseline Comparison Methods
```bash
# 0-shot (no examples)
python comparison_infer/task2_baseline_0shot. py

# 10-shot (1 example per class, 11 total)
python comparison_infer/task2_baseline_10shot.py
```

## 🔬 Experimental Comparison

This project compares the performance of three methods:

| Method | Description | Task 1 Script | Task 2 Script |
|--------|-------------|---------------|---------------|
| **0-shot** | Original model, no examples | `task1_baseline_0shot.py` | `task2_baseline_0shot.py` |
| **Few-shot (10-shot)** | Original model + balanced sampling examples (1 per class) | `task1_baseline_10shot. py` | `task2_baseline_10shot.py` |
| **SFT Fine-tuning** | Supervised fine-tuned model | `task1_inference.py` | `task2_inference. py` |

### Few-shot Strategy Description

The Few-shot baseline adopts a **balanced sampling strategy**:
- Select 1 sample from each category in the validation set as an example
- 11 categories in total, resulting in 11 examples
- Use a fixed random seed (seed=42) to ensure reproducibility

## 📈 Benchmark Testing

Run the complete benchmark test:
```bash
python training/run_benchmark. py
```

Visualize results:
```bash
python training/visualize_results.py
```

## 🔧 Configuration

| Configuration Item | Location | Description |
|--------------------|----------|-------------|
| Evaluation config | `eval/config.json` | Evaluation parameter settings |
| Prompt templates | `eval/prompt.json` | Prompt templates for each task |
| Model paths | Within each script | Modify according to actual environment |
| Demo config | `app.py` | Demo service parameter configuration |

## 📁 Output Files

### `output/` - Inference Prediction Results

| File | Description |
|------|-------------|
| `task1_predictions.jsonl` | Task 1 SFT fine-tuned model predictions |
| `task1_baseline_0shot_predictions.jsonl` | Task 1 0-shot baseline predictions |
| `task1_baseline_10shot_predictions. jsonl` | Task 1 10-shot baseline predictions |
| `task2_predictions.jsonl` | Task 2 SFT fine-tuned model predictions |
| `task2_baseline_0shot_predictions.jsonl` | Task 2 0-shot baseline predictions |
| `task2_baseline_10shot_predictions. jsonl` | Task 2 10-shot baseline predictions |

#### Prediction Result Format

**Task 1 (Intent Classification) Prediction Results:**
```json
{
  "instruction": "User input text",
  "predicted_category": "ORDER",
  "ground_truth_category": "ORDER",
  "ground_truth_response": "Expected customer service response..."
}
```

**Task 2 (Response Generation) Prediction Results:**
```json
{
  "instruction": "User input text",
  "category": "ORDER",
  "generated_response": "Model generated response...",
  "ground_truth_response": "Expected customer service response..."
}
```

### `checkpoint/` - Model Checkpoints

> ⚠️ **Note**: Due to the large size of checkpoint files, they are not uploaded to the GitHub repository. They will be automatically generated when running training scripts locally. 

| Directory | Description |
|-----------|-------------|
| `task1_classification/` | Intent classification task fine-tuned model weights |
| `task2_response_generation/` | Response generation task fine-tuned model weights |

### `models/` - Pre-trained Base Models

> ⚠️ **Note**: Due to the large size of model files, they are not uploaded to the GitHub repository. Please download the base model manually and place it in this directory.

| Directory | Description |
|-----------|-------------|
| `Qwen2.5-1.5B-Instruct/` | Qwen2. 5 1.5B Instruct base model |

### Other Output Directories

| Directory | Content |
|-----------|---------|
| `eval_output/` | Evaluation results and metrics |
| `benchmark_output/` | Benchmark test results |

## 🌐 Demo

This project provides a complete Web Demo for an intuitive experience of the intelligent email assistant functionality.

### Features

- 🏷️ **Email Classification**: Automatically identify the intent category of user emails
- 💬 **Response Generation**: Generate professional customer service responses based on email content and classification
- 🔄 **One-click Processing**: Support simultaneous classification and response generation

### Start Demo

```bash
# Install additional dependencies
pip install flask flask-cors

# Start service
python app. py
```

### Access URLs

After the service starts, you can access it through the following methods:

| Access Method | URL |
|---------------|-----|
| Local access | http://localhost:5000 |
| LAN access | http://<server-IP>:5000 |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Frontend page |
| GET | `/api/health` | Health check |
| POST | `/api/classify` | Email classification |
| POST | `/api/generate` | Response generation |
| POST | `/api/process` | Complete processing (classification + generation) |

#### API Request Examples

**Email Classification `/api/classify`**
```json
// Request
{"email": "I want to cancel my subscription"}

// Response
{"success": true, "category": "CANCEL"}
```

**Response Generation `/api/generate`**
```json
// Request
{"email": "I want to cancel my subscription", "category": "CANCEL"}

// Response
{"success": true, "response": "We're sorry to hear that you want to cancel... "}
```

**Complete Processing `/api/process`**
```json
// Request
{"email": "I want to cancel my subscription"}

// Response
{
  "success": true,
  "category": "CANCEL",
  "response": "We're sorry to hear that you want to cancel..."
}
```

### Demo Configuration

Demo service configuration parameters are located at the top of `app.py`:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `HOST` | `0.0.0.0` | Service listening address |
| `PORT` | `5000` | Service port |
| `BASE_MODEL_PATH` | `models/Qwen2.5-1. 5B-Instruct/` | Qwen2.5 base model path |
| `TASK1_CHECKPOINT` | `checkpoint/task1_classification/final_model` | Classification model path |
| `TASK2_CHECKPOINT` | `checkpoint/task2_response_generation/final_model` | Generation model path |

## 📄 License

This project is for academic research purposes only. 

## 👥 Authors

AIAA3102 Course Student Project