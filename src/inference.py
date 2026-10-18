"""
Inference Engine and Standout Production Interface for SentimentScope.
Provides the SentimentAnalyzer class for single and batch review sentiment analysis.
"""

import os
import torch
from transformers import AutoTokenizer
from .model import DemoGPT


class SentimentAnalyzer:
    """
    Production-ready inference interface for CineScope sentiment analysis.
    Loads trained DemoGPT checkpoint and predicts sentiment on single/batch inputs.
    """
    def __init__(self, checkpoint_path=None, tokenizer_name="bert-base-uncased", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.default_config = {
            "vocabulary_size": self.tokenizer.vocab_size,
            "num_classes": 2,
            "d_embed": 128,
            "context_size": 128,
            "layers_num": 4,
            "heads_num": 4,
            "head_size": 32,
            "dropout_rate": 0.1,
            "use_bias": True
        }

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            self.model = DemoGPT(self.default_config).to(self.device)
            self.model.eval()

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", self.default_config) if isinstance(checkpoint, dict) else self.default_config
        self.model = DemoGPT(config).to(self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            # direct state_dict
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        # Older checkpoints named the head `classification_head` and had no bias;
        # map it onto `classifier` with a zero bias (numerically identical).
        state_dict = dict(state_dict)
        if "classification_head.weight" in state_dict:
            state_dict["classifier.weight"] = state_dict.pop("classification_head.weight")
            if "classifier.bias" not in state_dict and self.model.classifier.bias is not None:
                state_dict["classifier.bias"] = torch.zeros_like(self.model.classifier.bias)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.config = config

    def predict(self, texts, max_length=128):
        """
        Predict sentiment for a single string or list of strings.

        Args:
            texts (str or list of str): Input review(s)
            max_length (int): Maximum sequence token length
        Returns:
            list of dict: Detailed prediction for each review.
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        self.model.eval()
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        results = []
        for i, text in enumerate(texts):
            pred_label = predictions[i].item()
            probs = probabilities[i].tolist()
            confidence = probs[pred_label]
            sentiment_label = "Positive" if pred_label == 1 else "Negative"

            results.append({
                "text": text,
                "label": pred_label,
                "sentiment": sentiment_label,
                "confidence": confidence,
                "probabilities": {
                    "negative": probs[0],
                    "positive": probs[1]
                }
            })

        return results[0] if is_single else results

    def format_predictions(self, results):
        """
        Format prediction dictionary for CLI or report display.
        """
        if isinstance(results, dict):
            results = [results]

        output_lines = []
        for idx, item in enumerate(results, 1):
            icon = "👍 Positive" if item["label"] == 1 else "👎 Negative"
            conf = item["confidence"] * 100
            preview = (item["text"][:80] + "...") if len(item["text"]) > 80 else item["text"]
            output_lines.append(f'[{idx}] Review: "{preview}"')
            output_lines.append(f"    Sentiment: {icon} (Confidence: {conf:.2f}%)")
            output_lines.append(f"    P(Neg): {item['probabilities']['negative']:.4f} | P(Pos): {item['probabilities']['positive']:.4f}\n")
        return "\n".join(output_lines)
