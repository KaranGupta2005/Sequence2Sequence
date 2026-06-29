"""
Training script for the improved Seq2Seq model.
Usage: python train.py --data data/fra.txt --epochs 15
"""

import argparse
import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from model import Encoder, Attention, Decoder, Seq2Seq
from vocab import clean_text, tokenize, build_vocab, encode, load_pairs, save_vocab


class TranslationDataset(Dataset):
    def __init__(self, pairs, fr_vocab, en_vocab):
        self.pairs = pairs
        self.fr_vocab = fr_vocab
        self.en_vocab = en_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fr_sentence, en_sentence = self.pairs[idx]
        src = torch.tensor(encode(fr_sentence, self.fr_vocab), dtype=torch.long)
        trg = torch.tensor(encode(en_sentence, self.en_vocab), dtype=torch.long)
        return src, trg


def collate_fn(batch):
    src, trg = zip(*batch)
    return pad_sequence(src, batch_first=True, padding_value=0), pad_sequence(trg, batch_first=True, padding_value=0)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    pairs = load_pairs(args.data)
    print(f"Total pairs: {len(pairs)}")

    if args.max_length:
        pairs = [(fr, en) for fr, en in pairs if len(fr.split()) <= args.max_length and len(en.split()) <= args.max_length]
        print(f"Filtered (max {args.max_length} words): {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)

    # Build vocab
    fr_vocab = build_vocab([p[0] for p in pairs], min_freq=args.min_freq)
    en_vocab = build_vocab([p[1] for p in pairs], min_freq=args.min_freq)
    inv_en_vocab = {v: k for k, v in en_vocab.items()}
    print(f"FR vocab: {len(fr_vocab)}, EN vocab: {len(en_vocab)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_vocab(fr_vocab, os.path.join(args.output_dir, "fr_vocab.json"))
    save_vocab(en_vocab, os.path.join(args.output_dir, "en_vocab.json"))

    config = {
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "bidirectional": True,
        "fr_vocab_size": len(fr_vocab),
        "en_vocab_size": len(en_vocab),
        "max_length": args.max_length or 50,
    }
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Data loaders
    val_size = int(len(pairs) * 0.05)
    train_ds = TranslationDataset(pairs[val_size:], fr_vocab, en_vocab)
    val_ds = TranslationDataset(pairs[:val_size], fr_vocab, en_vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    encoder = Encoder(len(fr_vocab), args.embed_dim, args.hidden_dim, args.n_layers, args.dropout)
    attention = Attention(args.hidden_dim)
    decoder = Decoder(len(en_vocab), args.embed_dim, args.hidden_dim, attention, args.n_layers, args.dropout)
    model = Seq2Seq(encoder, decoder, len(en_vocab), device).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0, 0
        tf = max(0.5, 1.0 - epoch * 0.05)
        t0 = time.time()

        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            out = model(src, trg, tf)
            loss = criterion(out[:, 1:].reshape(-1, out.shape[-1]), trg[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

        # Validation
        model.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                out = model(src, trg, 0.0)
                val_loss += criterion(out[:, 1:].reshape(-1, out.shape[-1]), trg[:, 1:].reshape(-1)).item()
                vn += 1

        vl = val_loss / vn
        scheduler.step(vl)
        saved = ""
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            saved = " *BEST*"

        print(f"Epoch {epoch+1}/{args.epochs} | Train: {total_loss/n:.4f} | Val: {vl:.4f} | TF: {tf:.2f} | {time.time()-t0:.1f}s{saved}")

    print(f"\nDone! Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/fra.txt")
    parser.add_argument("--output_dir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=20)
    args = parser.parse_args()
    train(args)
