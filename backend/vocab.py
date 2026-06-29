"""
Vocabulary building and text processing utilities for the Seq2Seq model.
"""

import re
import json
from collections import Counter
from typing import List, Tuple, Dict


def clean_text(text: str) -> str:
    """Clean and normalize text for the model."""
    text = text.lower()
    text = re.sub(r"[^a-zA-ZàâäéèêëïîôùûüÿçœæÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ0-9?.!,'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
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
    for word, freq in counter.items():
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
