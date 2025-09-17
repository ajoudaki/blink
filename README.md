# 🧠 Decoding Subjectivity: An AI That Learns Human Taste

What makes a face attractive, trustworthy, or smart? The answer is deeply personal and different for everyone. This project introduces a framework to teach an AI to understand these nuanced, subjective human perceptions.

We go beyond generic models by creating **personalized perception engines**. Using a rich dataset of human judgments and a powerful unified architecture, our models can learn an individual's unique "taste fingerprint" and use it to predict their preferences with high accuracy.

This repository contains everything you need: the complete dataset, sophisticated data analysis tools, and the code to train state-of-the-art models that can predict subjective traits from a face.

---

## ✨ Key Features

* **Personalized AI:** Train models that understand the preferences of a *specific user*, not just the average opinion.
* **High Accuracy:** Achieves **72.3% accuracy** in predicting which face a person will find more attractive, trustworthy, or smart in a pairwise comparison.
* **Dual Architectures:**
    1.  **High-Performance MLP:** A deep, optimized network for maximum accuracy on general tasks.
    2.  **User-Specific LoRA:** A memory-efficient fine-tuning approach to adapt a large CLIP model to a single user's taste.
* **Rich Subjectivity Dataset:** The engine is powered by nearly **20,000 pairwise comparisons** and **15,000 individual ratings** on three key traits.
* **Rigorous Quality Control:** Includes advanced analytics to ensure the human-provided data is consistent and reliable.

---

## 🚀 Quick Start

Get up and running in minutes. First, install the dependencies:

```bash
pip install torch torchvision pandas scikit-learn hydra-core matplotlib tqdm clip-by-openai
````

### Train the Best Models

This repo uses a unified training script. Here's how to train the two flagship models:

```bash
# 1. Train the highest-accuracy traditional model (72.3%)
# This uses our optimized MLP architecture.
python train.py --config-path configs --config-name unified_optimal_final

# 2. Train a memory-efficient, user-specific model with LoRA
# This fine-tunes a CLIP model for hyper-personalization.
python train_lora.py --config-name lora_single_user
```

> **Note:** Customize any run easily\! For example, to train a rating model with user encodings for 20 epochs:
> `python train.py task_type=rating training.epochs=20 model.use_user_encoding=true`

-----

## 🤖 Model Showdown: Traditional vs. LoRA

We offer two powerful paths to model subjectivity, each with unique advantages.

| Method | Script | Config | Test Accuracy | Key Advantage |
|--------|--------|--------|---------------|---------------|
| **Traditional MLP** | `train.py` | `unified_optimal_final` | **72.34%** | 🏆 **Highest accuracy** and stable, optimized training. |
| **LoRA Fine-tuning** | `train_lora.py`| `lora_single_user`| *Validated* | ⚡️ **Hyper-personalization**. Memory-efficient and perfect for creating custom models for many individual users. |

### The Unified Architecture

The magic lies in a flexible backbone that processes a **CLIP image embedding** and an optional **user ID**. This shared core feeds into different "heads" depending on the task, allowing for fair comparisons and easy experimentation.

  * **Rating Task:** Predicts a score from 1-4.
  * **Comparison Task:** Decides which of two images better fits a trait.

-----

## 📊 The Dataset: The Fuel for Perception

Our models are trained on a comprehensive dataset of human perceptions of FFHQ images.

  * **Total Individual Ratings:** \~14,579
  * **Total Pairwise Comparisons:** \~19,854
  * **Unique Images:** \~5,002
  * **Attributes:** Attractiveness, Smartness, Trustworthiness

The data is meticulously organized into individual ratings, pairwise comparisons, and image metadata. You can find the detailed schema and file descriptions in the `data/` directory.

### ✅ Behind the Scenes: Ensuring Data Quality

Garbage in, garbage out. That's why we've built a suite of powerful analysis tools (`analysis/`) to ensure our human labels are reliable. We automatically assess:

  * **Consistency:** Do a user's pairwise choices match their individual ratings?
  * **Random Behavior:** Is a user clicking too fast or inconsistently?
  * **Predictability:** Can we model a user's rating patterns, or are they an outlier?

This rigorous process guarantees that our models learn from high-quality, meaningful data.

-----

## 📁 Project Structure

The repository is organized for clarity and reproducibility.

```
.
├── configs/          # Hydra configuration files for all experiments ⚙️
├── data/             # The raw dataset files (.xlsx) 📦
├── runs/             # Outputs from every training run (logs, results, configs) 📈
├── artifacts/        # Persistent files like cached embeddings and saved models 💾
├── analysis/         # Jupyter notebooks and scripts for data quality analysis 🔬
├── train.py          # Main script for traditional MLP training
└── train_lora.py     # Script for LoRA fine-tuning
```

Each training run in `runs/` is timestamped and self-contained, including its configuration, CSV results, and a quick summary, making every experiment **100% reproducible**.

For a deeper dive into data schemas, model architectures, and analysis methods, please refer to the extended documentation sections. *[You can append the original, more detailed sections here if desired].*
