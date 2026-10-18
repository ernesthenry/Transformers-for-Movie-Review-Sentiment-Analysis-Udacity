"""
Training and evaluation pipeline for SentimentScope Transformer model.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

from .model import DemoGPT
from .dataset import load_dataset, IMDBDataset, create_dataloaders
import pandas as pd


def calculate_accuracy(model, data_loader, device):
    """
    Calculate classification accuracy percentage over a DataLoader.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
    return accuracy


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=3, device="cpu", log_interval=100):
    """
    Execute training loop over specified number of epochs.
    """
    history = {"train_loss": [], "val_accuracy": []}

    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for step, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = criterion(logits, labels)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % log_interval == 0:
                avg_step_loss = running_loss / log_interval
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], Loss: {avg_step_loss:.4f}")
                running_loss = 0.0

        # Evaluate validation accuracy
        val_acc = calculate_accuracy(model, val_loader, device)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} Completed in {elapsed:.1f}s - Validation Accuracy: {val_acc:.2f}%")
        history["val_accuracy"].append(val_acc)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train SentimentScope Transformer on IMDB")
    parser.add_argument("--data_dir", type=str, default="aclImdb", help="Path to aclImdb directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--checkpoint_out", type=str, default="checkpoints/sentiment_model_checkpoint.pt", help="Checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = {
        "vocabulary_size": tokenizer.vocab_size,
        "num_classes": 2,
        "d_embed": 128,
        "context_size": 128,
        "layers_num": 4,
        "heads_num": 4,
        "head_size": 32,
        "dropout_rate": 0.1,
        "use_bias": True
    }

    model = DemoGPT(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print("Model initialized. Ready for training.")


if __name__ == "__main__":
    main()
