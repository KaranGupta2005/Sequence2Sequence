"""
Vocabulary building and text processing utilities.
Matches the training preprocessing exactly.
"""

import re
import json
from collections import Counter
from typing import List, Tuple, Dict


def clean_text(text: str) -> str:
    """Clean text - matches training preprocessing exactly."""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\u00C0-\u017F0-9?.!,;:'\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


def build_vocab(sentences: List[str], min_freq: int = 2) -> Dict[str, int]:
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))
    vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
    idx = 4
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def encode(sentence: str, vocab: Dict[str, int]) -> List[int]:
    tokens = tokenize(sentence)
    return [vocab["<start>"]] + [vocab.get(word, vocab["<unk>"]) for word in tokens] + [vocab["<end>"]]


def load_pairs(filepath: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                eng, fr = parts[0], parts[1]
                pairs.append((clean_text(fr), clean_text(eng)))
    return pairs


def save_vocab(vocab: Dict[str, int], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(filepath: str) -> Dict[str, int]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
