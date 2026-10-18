# 🎬 SentimentScope: Transformer-based Sentiment Analysis for CineScope

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-76.05%25-brightgreen?style=for-the-badge)](https://github.com/ernesthenry/SentimentScope)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A complete Transformer Encoder designed from first principles with sequence mean pooling and classification head for binary sentiment analysis on IMDB movie reviews.**

*Built for CineScope's Content Recommendation and Personalization Engine.*

[Overview](#-overview) • [Architecture](#-architecture) • [Results](#-results) • [Standout Features](#-standout-features) • [Quick Start](#-quick-start) • [Project Structure](#-project-structure) • [Rubric Checklist](#-rubric-verification)

</div>

---

## 📋 Overview

At **CineScope**, understanding viewer reception is fundamental to recommending movies and television series that align with user preferences. Traditional collaborative filtering struggles with newly released media and unrated catalog titles. 

**SentimentScope** addresses this cold-start and engagement challenge by training a custom Transformer-based neural network (`DemoGPT`) from scratch to perform sentiment analysis on 50,000 real-world reviews from the IMDB dataset. By classifying raw reviews into positive or negative feedback, SentimentScope provides actionable sentiment scores that directly fuel CineScope's recommendation algorithms.

### Key Objectives Achieved
- ✅ Loaded, explored, and visualized 50,000 reviews with descriptive statistics and length distributions.
- ✅ Implemented a custom PyTorch `Dataset` with `bert-base-uncased` subword tokenization, padding, and truncation.
- ✅ Customized a Transformer architecture for sequence classification using **mean pooling** across the temporal dimension and a linear classification head.
- ✅ Implemented `calculate_accuracy()` and an epoch-based training loop with `AdamW` and `CrossEntropyLoss`.
- ✅ Exceeded the project benchmark threshold of **> 75% test accuracy** on unseen data (**76.05%** achieved).
- ✅ **Standout Extensions**: Persisted model checkpoint (`sentiment_model_checkpoint.pt`), built an object-oriented batch inference engine (`SentimentAnalyzer`), and authored an in-depth engineering report (`PROJECT_REPORT.md`).

---

## 🏗 Architecture

### From Generative Decoder to Discriminative Classifier
Unlike autoregressive language models that predict the next token one-by-one:
1. **Subword Tokenizer**: Uses `bert-base-uncased` WordPiece vocabulary (30,522 tokens) for efficient handling of compound words and movie terminology.
2. **Transformer Blocks**: 4 stacked blocks featuring multi-head self-attention, layer normalization, residual skip connections, and GELU feed-forward networks.
3. **Temporal Mean Pooling**: Condenses token vectors across the sequence length into a single fixed-length embedding $\mathbf{h}_{	ext{pooled}} = rac{1}{T}\sum_{t=1}^T \mathbf{h}_t \in \mathbb{R}^{128}$.
4. **Classification Head**: Linear layer mapping from $\mathbb{R}^{128} 	o \mathbb{R}^2$ producing logits for Negative (0) and Positive (1) classes.

```
Input Tokens (B, 128) 
        ↓
[Token Embedding] + [Positional Embedding]  → (B, 128, 128)
        ↓
[Transformer Block 1: MHA + FFN + Residuals + LN]
        ↓
[Transformer Block 2: MHA + FFN + Residuals + LN]
        ↓
[Transformer Block 3: MHA + FFN + Residuals + LN]
        ↓
[Transformer Block 4: MHA + FFN + Residuals + LN]
        ↓
[Layer Normalization]                       → (B, 128, 128)
        ↓
[Mean Pooling across Time dim=1]            → (B, 128)
        ↓
[Linear Classifier: 128 -> 2]               → (B, 2) [Logits]
```

### Model Configuration

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| `vocabulary_size` | 30,522 | BERT WordPiece subword dictionary size |
| `d_embed` | 128 | Embedding and latent feature dimension |
| `context_size` | 128 | Standardized token sequence length |
| `layers_num` | 4 | Number of stacked Transformer blocks |
| `heads_num` | 4 | Multi-head attention heads per block |
| `head_size` | 32 | Dimension per attention head ($4 	imes 32 = 128$) |
| `dropout_rate` | 0.1 | Regularization dropout probability |
| `num_classes` | 2 | Binary classification (0: Negative, 1: Positive) |
| **Total Parameters** | **4,814,850** | ~4.81M trainable parameters |

---

## 📊 Results

### Validation Accuracy & Loss Progression

| Epoch | Step Range | Training Loss (Avg / 100 steps) | Validation Accuracy | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **Untrained** | - | - | **49.76%** | Baseline random guess |
| **Epoch 1** | 1 - 704 | 0.6927 $	o$ 0.5496 | **70.00%** | Rapid convergence |
| **Epoch 2** | 705 - 1408 | 0.5124 $	o$ 0.4763 | **74.08%** | Feature stabilization |
| **Epoch 3** | 1409 - 2112 | 0.4610 $	o$ 0.4237 | **77.12%** | Validation milestone |
| **Final Test**| **25,000 samples** | - | **76.05%** | **Rubric requirement passed (> 75%)** |

---

## 🌟 Standout Features

### 1. Serialized Model Checkpoint (`sentiment_model_checkpoint.pt`)
The trained weights, config dictionary, and benchmark metrics are saved and loadable:
- Checkpoint location: `checkpoints/sentiment_model_checkpoint.pt` and root `sentiment_model_checkpoint.pt` (~19 MB).
- Accompanied by metadata: `checkpoints/model_metadata.json`.

### 2. Standout Inference Interface (`SentimentAnalyzer`)
We created a clean, modular class in `src/inference.py` for batch inference, probability calculation, and human-readable reporting:

```python
from src.inference import SentimentAnalyzer

analyzer = SentimentAnalyzer(checkpoint_path="sentiment_model_checkpoint.pt")
reviews = [
    "This movie was an absolute masterpiece! Brilliant acting and breathtaking visuals.",
    "Terrible waste of time. The plot was boring and made no sense at all."
]

predictions = analyzer.predict(reviews)
print(analyzer.format_predictions(predictions))
```

**Output:**
```text
[1] Review: "This movie was an absolute masterpiece! Brilliant acting and breathtaking visuals."
    Sentiment: 👍 Positive (Confidence: 99.77%)
    P(Neg): 0.0023 | P(Pos): 0.9977

[2] Review: "Terrible waste of time. The plot was boring and made no sense at all."
    Sentiment: 👎 Negative (Confidence: 98.64%)
    P(Neg): 0.9864 | P(Pos): 0.0136
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ernesthenry/SentimentScope.git
cd SentimentScope
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook SentimentScope.ipynb
```

### 3. Run Inference via Python CLI
```python
from src.inference import SentimentAnalyzer

analyzer = SentimentAnalyzer("sentiment_model_checkpoint.pt")
print(analyzer.predict("Inception is an extraordinary, mind-bending cinematic adventure!"))
```

---

## 📁 Project Structure

```
SentimentScope/
├── SentimentScope.ipynb          # Complete, fully executed submission notebook
├── PROJECT_REPORT.md             # In-depth engineering report and key takeaways
├── README.md                     # Project documentation and architecture guide
├── requirements.txt              # Environment dependencies
├── sentiment_model_checkpoint.pt # Standout trained model checkpoint (~19 MB)
├── checkpoints/
│   ├── sentiment_model_checkpoint.pt
│   └── model_metadata.json       # Training hyperparameters & metrics
└── src/
    ├── __init__.py               # Package exports
    ├── model.py                  # DemoGPT, AttentionHead, MultiHeadAttention, Block
    ├── dataset.py                # IMDBDataset and load_dataset helper
    ├── train.py                  # Training loop & calculate_accuracy utility
    └── inference.py              # Standout SentimentAnalyzer inference engine
```

---

## 📝 Rubric Verification

| Rubric Criteria | Requirement | Status |
| :--- | :--- | :---: |
| **Load Data** | `load_dataset()` loads 25k train and 25k test reviews with correct shapes `(25000, 2)`. | ✅ Passed |
| **Explore Data** | Descriptive statistics calculated; 2+ visualizations with proper labels plotted. | ✅ Passed |
| **Custom Dataset** | `IMDBDataset` implements `__init__`, `__len__`, `__getitem__`; asserts verify lengths & types. | ✅ Passed |
| **Customize Architecture**| `DemoGPT` implements mean pooling and linear head; output matches `(BATCH_SIZE, 2)`. | ✅ Passed |
| **Accuracy Function** | `calculate_accuracy()` accurately computes classification accuracy on PyTorch tensors. | ✅ Passed |
| **Training Loop** | Model trains without errors; validation accuracy tracked across epochs. | ✅ Passed |
| **Test Performance** | Model achieves **> 75%** test accuracy (**76.05%** achieved on 25k test reviews). | ✅ Passed |
| **Project Report** | Results summarized with 2+ key takeaways in `PROJECT_REPORT.md` and notebook. | ✅ Passed |
| **Standout Features** | Model checkpoint saved and simple inference interface provided for batch predictions. | ✅ Passed |

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
