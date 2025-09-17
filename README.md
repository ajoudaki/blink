
# 🧠 Decoding Subjectivity: An AI That Learns Human Taste

What makes a face attractive, trustworthy, or smart? The answer is deeply personal and different for everyone. This project introduces a framework to teach an AI to understand these nuanced, subjective human perceptions.

We go beyond generic models by creating **personalized perception engines**. Using a rich dataset of human judgments and two powerful architectures, our models can learn an individual's unique "taste fingerprint" and use it to predict their preferences with remarkable accuracy.

This repository contains everything you need: the complete dataset, sophisticated data analysis tools, and the code to train and compare these state-of-the-art models.

---

## ✨ Key Features

* **Personalized AI:** Train models that understand the preferences of a *specific user*, not just the average opinion.
* **State-of-the-Art Accuracy:** Our models achieve over **72% accuracy** in predicting subjective human choices.
* **Two Powerful Architectures:**
    1.  **MLP Decoder:** A powerful, multi-headed decoder built on a **frozen CLIP model** for high performance.
    2.  **LoRA Fine-tuning:** An elegant, parameter-efficient approach that fine-tunes CLIP and learns **specialized user tokens**, unlocking zero-shot potential for new questions.
* **Extreme Personalization:** For some users, the personalized model's accuracy soars as high as **87%**, demonstrating the power of user-specific adaptation.
* **Rich Subjectivity Dataset:** The engine is powered by nearly **20,000 pairwise comparisons** and **15,000 individual ratings** on three key traits.

---

## 🚀 Quick Start

Get up and running in minutes. First, install the dependencies:

```bash
pip install torch torchvision pandas scikit-learn hydra-core matplotlib tqdm clip-by-openai
````

### Train the Best Models

Here's how to train our two flagship models:

```bash
# 1. Train the high-performance MLP Decoder model (on a frozen CLIP)
python train.py --config-path configs --config-name unified_optimal_final

# 2. Train the elegant LoRA Fine-tuning model
# This approach fine-tunes CLIP itself and has zero-shot potential.
python train_lora.py --config-name lora_single_user
```

-----

## 🤖 Two Roads to Personalization: MLP Decoder vs. LoRA Fine-tuning

We offer two distinct, powerful architectures for modeling subjectivity. While their performance is competitive, their philosophies and capabilities differ significantly.

| Aspect | MLP Decoder Approach | LoRA Fine-tuning Approach |
| :--- | :--- | :--- |
| **CLIP Model State** | 🧊 **Completely Frozen** | 🔥 **Fine-Tuned** (with LoRA) |
| **Personalization** | One-Hot User ID → **Complex MLP Head** | **Learned User Token** → Simple Cosine Sim. |
| **Task Handling** | Separate output head per attribute | Text-based (e.g., "attractive"), enabling **zero-shot** |
| **Key Advantage** | High performance with a simple base | More elegant, uniform, and extensible design |

### 1\. The MLP Decoder: Power on a Frozen Foundation

This method keeps the massive CLIP vision model completely untouched. All the learning happens in a sophisticated MLP "decoder" that sits on top.

  * **How it works:** It takes the CLIP image embedding and a one-hot vector for the user ID, and a powerful MLP learns to map that combination to a preference score for each attribute (attractiveness, smartness, etc.) through separate heads.
  * **Pros:** Very high performance, conceptually straightforward.
  * **Cons:** Less flexible; requires a new output head for every new question you want to ask.

### 2\. LoRA Fine-tuning: The Elegant, Zero-Shot Future

This method is more surgical. Instead of adding a big decoder, it slightly modifies the CLIP model itself using efficient LoRA adapters and learns a unique "user token" that represents each person's taste.

  * **How it works:** The model learns to encode a user's preference into a special token. To make a prediction, it simply compares the image to the text of the question (e.g., "a photo of an attractive person") within CLIP's embedding space, guided by the user token.
  * **Pros:** More elegant and uniform. Has the potential to answer **new questions in a zero-shot fashion** because it relies on CLIP's natural language understanding.
  * **Cons:** Performance is highly competitive but requires tuning the core CLIP model.

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
├── train.py          # Main script for the MLP Decoder approach
└── train_lora.py     # Script for the LoRA Fine-tuning approach
```

Each training run in `runs/` is timestamped and self-contained, including its configuration, CSV results, and a quick summary, making every experiment **100% reproducible**.
