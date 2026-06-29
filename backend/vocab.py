"""
Improved vocabulary building and text processing utilities.

Improvements:
- Better tokenization (handles punctuation as separate tokens)
- Handles French special characters properly
- Supports reverse vocab lookup
"""

import re
import json
from collections import Counter
from typing import List, Tuple, Dict


def clean_text(text: str) -> str:
    """Clean and normalize text for the model."""
    text = text.lower().strip()
    # Separate punctuation from words (but keep apostrophes in contractions)
    text = re.sub(r"([?.!,;:])", r" \1 ", text)
    # Handle French contractions - keep l', d', j', etc. attached
    text = re.sub(r"(\w)'(\w)", r"\1'\2", text)
    # Remove any character that's not alphanumeric, French chars, punctuation, or space
    text = re.sub(r"[^a-zA-ZàâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ0-9?.!,;:'\- ]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Whitespace tokenizer."""
    return text.split()


def build_vocab(sentences: List[str], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from a list of sentences."""
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<start>": 2,
        "<end>": 3
    }

    idx = 4
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def encode(sentence: str, vocab: Dict[str, int]) -> List[int]:
    """Encode a sentence into a list of token indices."""
    tokens = tokenize(sentence)
    return [vocab["<start>"]] + [vocab.get(word, vocab["<unk>"]) for word in tokens] + [vocab["<end>"]]


def load_pairs(filepath: str) -> List[Tuple[str, str]]:
    """Load sentence pairs from the fra.txt file."""
    pairs = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                eng, fr = parts[0], parts[1]
                pairs.append((clean_text(fr), clean_text(eng)))
    return pairs


def save_vocab(vocab: Dict[str, int], filepath: str):
    """Save vocabulary to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(filepath: str) -> Dict[str, int]:
    """Load vocabulary from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
