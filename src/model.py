"""
Transformer Model Architecture for Binary Sentiment Classification (DemoGPT).
Implements custom multi-head self-attention, feed-forward sublayers, residual
connections with layer normalization, sequence mean pooling, and classification head.
"""

import math
import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """
    A single causal self-attention head.
    """
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config.get("use_bias", True))
        self.K_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config.get("use_bias", True))
        self.V_weights = nn.Linear(config["d_embed"], config["head_size"], bias=config.get("use_bias", True))

        self.dropout = nn.Dropout(config.get("dropout_rate", 0.1))

        casual_attention_mask = torch.tril(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer('casual_attention_mask', casual_attention_mask)

    def forward(self, input):
        batch_size, tokens_num, d_embed = input.shape
        Q = self.Q_weights(input)  # (B, T, head_size)
        K = self.K_weights(input)  # (B, T, head_size)
        V = self.V_weights(input)  # (B, T, head_size)

        # Scaled dot-product attention
        attention_scores = Q @ K.transpose(1, 2)  # (B, T, T)

        # Causal mask application
        attention_scores = attention_scores.masked_fill(
            self.casual_attention_mask[:tokens_num, :tokens_num] == 0,
            float('-inf')
        )
        attention_scores = attention_scores / math.sqrt(K.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        return attention_scores @ V  # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism projecting across multiple attention heads.
    """
    def __init__(self, config):
        super().__init__()
        heads_list = [AttentionHead(config) for _ in range(config["heads_num"])]
        self.heads = nn.ModuleList(heads_list)

        self.linear = nn.Linear(config["heads_num"] * config["head_size"], config["d_embed"])
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.1))

    def forward(self, input):
        heads_outputs = [head(input) for head in self.heads]
        x = torch.cat(heads_outputs, dim=-1)  # (B, T, heads_num * head_size)
        x = self.linear(x)                   # (B, T, d_embed)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU non-linearity and dropout.
    """
    def __init__(self, config):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config.get("dropout_rate", 0.1))
        )

    def forward(self, input):
        return self.linear_layers(input)


class Block(nn.Module):
    """
    Transformer Block consisting of pre-layer-normalized multi-head attention
    and feed-forward layers with residual skip connections.
    """
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(config["d_embed"])

        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(config["d_embed"])

    def forward(self, input):
        x = input
        x = x + self.multi_head(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class DemoGPT(nn.Module):
    """
    DemoGPT: Transformer-based Model customized for Binary Classification.
    
    Instead of auto-regressive next-token generation, this architecture:
    1. Embeds tokens and positions.
    2. Passes representations through stacked Transformer Blocks.
    3. Normalizes representations via LayerNorm.
    4. Condenses token vectors into a single sequence vector via Mean Pooling.
    5. Maps pooled embeddings to binary class logits via a linear classification head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding layer: Maps token indices to embedding vectors
        self.token_embedding_layer = nn.Embedding(config["vocabulary_size"], config["d_embed"])

        # Positional embedding layer: Adds positional coordinate information
        self.positional_embedding_layer = nn.Embedding(config["context_size"], config["d_embed"])

        # Transformer layers: Stack of transformer blocks
        blocks = [Block(config) for _ in range(config["layers_num"])]
        self.layers = nn.Sequential(*blocks)

        # Layer normalization to stabilize representation distribution
        self.layer_norm = nn.LayerNorm(config["d_embed"])

        # Classification output layer: Maps pooled embeddings to 2 class logits
        self.classifier = nn.Linear(config["d_embed"], config["num_classes"], bias=config.get("use_bias", True))

    def forward(self, token_ids):
        """
        Forward pass for binary sentiment classification.

        Args:
            token_ids (torch.Tensor): Token indices of shape (B, T)
        Returns:
            logits (torch.Tensor): Output class logits of shape (B, num_classes)
        """
        batch_size, tokens_num = token_ids.shape

        # Step 1: Token & Positional Embeddings
        x = self.token_embedding_layer(token_ids)  # (B, T, d_embed)
        positions = torch.arange(tokens_num, device=token_ids.device)  # (T,)
        pos_embed = self.positional_embedding_layer(positions)  # (T, d_embed)
        x = x + pos_embed.unsqueeze(0)  # (B, T, d_embed)

        # Step 2: Transformer Blocks
        x = self.layers(x)  # (B, T, d_embed)
        x = self.layer_norm(x)  # (B, T, d_embed)

        # Step 3: Mean pooling across sequence dimension (dim=1)
        x = torch.mean(x, dim=1)  # (B, d_embed)

        # Step 4: Classification head
        logits = self.classifier(x)  # (B, num_classes)

        return logits
