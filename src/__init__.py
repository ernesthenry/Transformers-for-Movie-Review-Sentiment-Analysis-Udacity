"""
SentimentScope: Transformer-based Sentiment Analysis Package
Built for CineScope Content Recommendation Engine
"""

from .model import DemoGPT, Block, MultiHeadAttention, AttentionHead, FeedForward
from .dataset import IMDBDataset, load_dataset
from .inference import SentimentAnalyzer

__version__ = '1.0.0'
__author__ = 'Kato Henry'
__all__ = [
    'DemoGPT',
    'Block',
    'MultiHeadAttention',
    'AttentionHead',
    'FeedForward',
    'IMDBDataset',
    'load_dataset',
    'SentimentAnalyzer'
]
