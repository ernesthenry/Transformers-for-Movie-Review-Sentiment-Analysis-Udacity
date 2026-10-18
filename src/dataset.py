"""
Dataset loading, preprocessing, and PyTorch Dataset definitions for IMDB sentiment data.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_dataset(folder):
    """
    Reads all text files in the specified folder and returns their content as a list.

    Args:
        folder (str): Path to folder containing review text files.
    Returns:
        list of str: Content of each review file.
    """
    reviews = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")
        
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                reviews.append(f.read())
    return reviews


class IMDBDataset(Dataset):
    """
    Custom PyTorch Dataset for IMDB Sentiment Classification.

    Tokenizes raw text using subword tokenizer (bert-base-uncased),
    pads/truncates to max_length, and returns PyTorch tensors with integer labels.
    """
    def __init__(self, data, tokenizer, max_length=128):
        """
        Args:
            data (pd.DataFrame): DataFrame with 'review' and 'label' columns.
            tokenizer: Hugging Face PreTrainedTokenizer (e.g. bert-base-uncased).
            max_length (int): Maximum token sequence length.
        """
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["review"])
        label = int(self.data.iloc[idx]["label"])

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        return input_ids, label


def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=32, max_length=128):
    """
    Create PyTorch DataLoaders for train, validation, and test splits.
    """
    train_ds = IMDBDataset(train_df, tokenizer, max_length=max_length)
    val_ds = IMDBDataset(val_df, tokenizer, max_length=max_length)
    test_ds = IMDBDataset(test_df, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
