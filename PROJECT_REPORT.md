# 📊 SentimentScope: Project Engineering & Evaluation Report

**Role**: Machine Learning Engineer  
**Organization**: CineScope Entertainment  
**Initiative**: Audience Sentiment Understanding for Recommendation Engine Refinement  
**Author**: Kato Henry  
**Date**: October 2026  

---

## 1. Executive Summary

CineScope delivers streaming entertainment discovery, helping viewers find shows and movies they love. To boost personalization and prevent negative recommendation churn, CineScope initiated **SentimentScope** — a deep learning pipeline tasked with deciphering fine-grained user sentiment from unstructured natural language reviews.

The objective was to design, construct, train, and validate a transformer-based neural network for binary sentiment classification (positive vs. negative) on the IMDB Large Movie Review benchmark (50,000 reviews). Rather than relying solely on pre-packaged high-level APIs, this project built a complete encoder architecture from first principles using PyTorch, adapted the generation-style attention mechanism for sequence classification via **mean pooling**, and achieved a verified test accuracy of **76.05%**, exceeding the project specification benchmark of > 75%.

---

## 2. Technical Architecture & Design Decisions

### 2.1 From Generative to Discriminative Transformers
Conventional GPT architectures perform auto-regressive next-token prediction, relying on triangular causal masking and projecting sequence hidden states to vocabulary distributions. Adapting this structure to binary classification required three fundamental engineering modifications:

1. **Subword Tokenization (`bert-base-uncased`)**:
   Instead of inefficient character-level tokenization or high-cardinality word-level dictionaries, we employed Hugging Face's WordPiece tokenizer with a 30,522-token vocabulary. Text sequences were truncated and padded to a standardized context size ($T = 128$), balancing computational throughput with linguistic coverage.

2. **Temporal Mean Pooling Layer**:
   Instead of extracting token-level predictions at each index, we implemented sequence-level pooling:
   $$\mathbf{h}_{	ext{pooled}} = rac{1}{T} \sum_{t=1}^{T} \mathbf{h}_t \in \mathbb{R}^{d_{	ext{embed}}}$$
   Mean pooling aggregates contextualized representations evenly across all active tokens, capturing broad semantic mood better than taking only the last token position.

3. **Classification Head**:
   A linear projection maps pooled embeddings $\mathbf{h}_{	ext{pooled}} \in \mathbb{R}^{128}$ directly to raw class logits:
   $$\mathbf{z} = \mathbf{W}_{	ext{class}} \mathbf{h}_{	ext{pooled}} + \mathbf{b} \in \mathbb{R}^2$$
   Class 0 corresponds to Negative sentiment; Class 1 corresponds to Positive sentiment.

### 2.2 Model Hyperparameter Configuration

```json
{
  "vocabulary_size": 30522,
  "num_classes": 2,
  "d_embed": 128,
  "context_size": 128,
  "layers_num": 4,
  "heads_num": 4,
  "head_size": 32,
  "dropout_rate": 0.1,
  "use_bias": true,
  "total_parameters": 4814850
}
```

---

## 3. Exploratory Data Analysis & Preparation

The dataset consists of 50,000 IMDB movie reviews split evenly between training and testing sets:

| Dataset Split | Positive Reviews | Negative Reviews | Total Records |
| :--- | :--- | :--- | :--- |
| **Train Set** | 12,500 (50.0%) | 12,500 (50.0%) | 25,000 |
| **Test Set** | 12,500 (50.0%) | 12,500 (50.0%) | 25,000 |

### Key EDA Observations:
1. **Balanced Class Distribution**: Exact 50/50 balance between positive and negative labels eliminates bias toward majority classes and validates accuracy as a primary metric.
2. **Review Length Variability**: Average review length was approximately 238 words (1,310 characters), with a maximum exceeding 2,400 words. Padded context size ($T=128$) captured the introductory and thesis paragraphs where users typically convey their verdict.
3. **Train-Validation Split**: The 25,000 training examples were shuffled with `random_state=42` and partitioned into:
   - **Training subset**: 22,500 reviews (90%)
   - **Validation subset**: 2,500 reviews (10%)

---

## 4. Training Dynamics & Experimental Results

The network was trained using the `AdamW` optimizer ($lpha = 3 	imes 10^{-4}$) and PyTorch `CrossEntropyLoss` with mini-batches of size 32.

### Training Progression Over Epochs

| Epoch | Step Range | Training Loss (Avg / 100 steps) | Validation Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: |
| **Untrained** | - | - | **49.76%** | Random chance |
| **1** | 1 - 704 | 0.6927 $	o$ 0.5496 | **70.00%** | Rapid feature learning |
| **2** | 705 - 1408 | 0.5124 $	o$ 0.4763 | **74.08%** | Attention alignment |
| **3** | 1409 - 2112 | 0.4610 $	o$ 0.4237 | **77.12%** | Benchmark exceeded (>75%) |
| **Final Test** | - | - | **76.05%** | **PASSED** |

### Test Performance Assessment
- **Final Test Accuracy**: **76.05%**
- **Rubric Minimum Target**: 75.00%
- **Result**: Successfully surpassed the threshold by +1.05% on unseen data.

---

## 5. Standout Engineering Features

To elevate SentimentScope beyond standard coursework requirements, two production-grade extensions were implemented:

### 5.1 Checkpoint Serialization & Model Persistence
- Full model state dictionary, architectural hyperparameters, and evaluation metrics were serialized to `checkpoints/sentiment_model_checkpoint.pt` and `sentiment_model_checkpoint.pt`.
- Accompanied by `checkpoints/model_metadata.json` for MLOps tracking.

### 5.2 Standout Reusable Inference Interface (`SentimentAnalyzer`)
An object-oriented inference engine (`src/inference.py`) that abstracts tokenization, tensor conversion, GPU/CPU dispatching, softmax probability scaling, and human-readable output formatting:

```python
from src.inference import SentimentAnalyzer

analyzer = SentimentAnalyzer(checkpoint_path="sentiment_model_checkpoint.pt")
results = analyzer.predict([
    "This movie was an absolute masterpiece! Brilliant acting and breathtaking visuals.",
    "Terrible waste of time. The plot was boring and made no sense at all."
])
print(analyzer.format_predictions(results))
```

**Output:**
```
[1] Review: "This movie was an absolute masterpiece! Brilliant acting and breathtaking visuals."
    Sentiment: 👍 Positive (Confidence: 99.77%)
    P(Neg): 0.0023 | P(Pos): 0.9977

[2] Review: "Terrible waste of time. The plot was boring and made no sense at all."
    Sentiment: 👎 Negative (Confidence: 98.64%)
    P(Neg): 0.9864 | P(Pos): 0.0136
```

---

## 6. Key Takeaways & Recommendations

1. **Mean Pooling vs. Causal Generation**:
   In autoregressive language generation, tokens only attend backward to avoid seeing future tokens, and output predictions are extracted from the sequence terminus. In discriminative classification, mean pooling aggregates representations across all positions, preserving distributed nuances and idiomatic cues across the entire review.

2. **Subword Tokenization Eliminates Vocabulary Sparsity**:
   Employing WordPiece (`bert-base-uncased`) allowed the model to handle domain jargon, misspelled actors' names, and rare movie titles without out-of-vocabulary degradation.

3. **Pathway to 90%+ Accuracy**:
   A from-scratch 4-layer transformer hits an empirical ceiling around 77–80% due to sample complexity on 22,500 training points. To reach > 90% accuracy in production, CineScope should adopt transfer learning with a pre-trained backbone (e.g. RoBERTa or DeBERTa) or unfreeze the top layers of BERT with cosine learning rate scheduling.

---

*Report certified by CineScope Machine Learning Engineering Team.*
