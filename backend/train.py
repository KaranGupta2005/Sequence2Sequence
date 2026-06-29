"""
Improved training script for the Seq2Seq French-to-English Translation model.

Improvements:
- Bidirectional Encoder + 2-layer LSTM
- Scheduled sampling (reduces teacher forcing over time)
- Learning rate scheduling
- Gradient clipping
- Train/Val split with early stopping
- BLEU score evaluation during training
- Better logging

Usage:
    python train.py --data data/fra.txt --epochs 15 --batch_size 64
"""

import argparse
import os
import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
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
    src_pad = pad_sequence(src, batch_first=True, padding_value=0)
    trg_pad = pad_sequence(trg, batch_first=True, padding_value=0)
    return src_pad, trg_pad


def compute_bleu_simple(references, hypotheses):
    """Simple BLEU-1 score for quick evaluation."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = set(ref.split())
        hyp_tokens = hyp.split()
        if not hyp_tokens:
            scores.append(0.0)
            continue
        matches = sum(1 for t in hyp_tokens if t in ref_tokens)
        precision = matches / len(hyp_tokens)
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref.split()) / max(len(hyp_tokens), 1)))
        scores.append(precision * bp)
    return sum(scores) / len(scores) if scores else 0.0


def translate_sentence(model, sentence, fr_vocab, en_vocab, inv_en_vocab, device, max_length=50, beam_width=3):
    """Translate a single sentence using beam search."""
    model.eval()
    cleaned = clean_text(sentence)
    src = torch.tensor(encode(cleaned, fr_vocab), dtype=torch.long).unsqueeze(0).to(device)

    tokens = model.beam_search(src, en_vocab["<start>"], en_vocab["<end>"], max_length, beam_width)

    # Filter out special tokens
    words = []
    for t in tokens:
        if t == en_vocab.get("<end>", -1):
            break
        word = inv_en_vocab.get(t, "<unk>")
        if word not in ("<start>", "<end>", "<pad>"):
            words.append(word)

    return " ".join(words)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            outputs = model(src, trg, teacher_forcing_ratio=0.0)
            outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(outputs, trg_flat)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"  Seq2Seq French→English Translation - Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load and preprocess data
    print(f"\n[1/5] Loading data...")
    pairs = load_pairs(args.data)
    print(f"  Total pairs loaded: {len(pairs)}")

    # Filter to shorter sentences
    if args.max_length:
        pairs = [(fr, en) for fr, en in pairs
                 if len(fr.split()) <= args.max_length and len(en.split()) <= args.max_length]
        print(f"  Filtered (max {args.max_length} words): {len(pairs)} pairs")

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)

    # Build vocabularies
    print(f"\n[2/5] Building vocabularies...")
    fr_sentences = [p[0] for p in pairs]
    en_sentences = [p[1] for p in pairs]

    fr_vocab = build_vocab(fr_sentences, min_freq=args.min_freq)
    en_vocab = build_vocab(en_sentences, min_freq=args.min_freq)
    inv_en_vocab = {v: k for k, v in en_vocab.items()}

    print(f"  French vocab: {len(fr_vocab)} tokens")
    print(f"  English vocab: {len(en_vocab)} tokens")

    # Save vocabularies
    os.makedirs(args.output_dir, exist_ok=True)
    save_vocab(fr_vocab, os.path.join(args.output_dir, "fr_vocab.json"))
    save_vocab(en_vocab, os.path.join(args.output_dir, "en_vocab.json"))

    # Save model config
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

    # Train/Val split
    print(f"\n[3/5] Creating data loaders...")
    val_size = int(len(pairs) * 0.05)
    train_pairs = pairs[val_size:]
    val_pairs = pairs[:val_size]

    train_dataset = TranslationDataset(train_pairs, fr_vocab, en_vocab)
    val_dataset = TranslationDataset(val_pairs, fr_vocab, en_vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    print(f"  Train: {len(train_pairs)} pairs ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_pairs)} pairs")

    # Create model
    print(f"\n[4/5] Creating model...")
    encoder = Encoder(len(fr_vocab), args.embed_dim, args.hidden_dim,
                      n_layers=args.n_layers, dropout=args.dropout)
    attention = Attention(args.hidden_dim)
    decoder = Decoder(len(en_vocab), args.embed_dim, args.hidden_dim, attention,
                      n_layers=args.n_layers, dropout=args.dropout)
    model = Seq2Seq(encoder, decoder, len(en_vocab), device).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Architecture: Bi-LSTM {args.n_layers}L, {args.embed_dim}emb, {args.hidden_dim}hid")

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

    # Training loop
    print(f"\n[5/5] Training for {args.epochs} epochs...")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        # Scheduled sampling: reduce teacher forcing over time
        tf_ratio = max(0.5, 1.0 - (epoch * 0.05))

        for batch_idx, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)

            outputs = model(src, trg, teacher_forcing_ratio=tf_ratio)
            outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        elapsed = time.time() - start_time

        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Sample translations
        print(f"\n  Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"TF: {tf_ratio:.2f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Show sample translations every few epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            samples = ["Bonjour.", "Merci beaucoup.", "Je suis un homme.", "Comment allez-vous?", "Je t'aime."]
            print(f"  {'─'*50}")
            for s in samples:
                t = translate_sentence(model, s, fr_vocab, en_vocab, inv_en_vocab, device)
                print(f"    {s} → {t}")
            model.train()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
            print(f"  ★ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {args.output_dir}/model.pt")
    print(f"{'='*60}")

    # Final evaluation with more samples
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt"), map_location=device))
    model.eval()

    print(f"\n  Final Translations:")
    print(f"  {'─'*50}")
    test_sentences = [
        "Bonjour.", "Merci beaucoup.", "Je t'aime.", "Comment allez-vous?",
        "Je suis un homme.", "Elle est belle.", "Il fait froid.",
        "Bonne nuit.", "Nous sommes amis.", "Je ne comprends pas.",
        "Il est tard.", "Je veux dormir.", "Tu es gentil.",
        "Je suis heureux.", "Au revoir.",
    ]
    for s in test_sentences:
        t = translate_sentence(model, s, fr_vocab, en_vocab, inv_en_vocab, device)
        print(f"    {s} → {t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Improved Seq2Seq French→English Model")
    parser.add_argument("--data", type=str, default="data/fra.txt", help="Path to fra.txt")
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--min_freq", type=int, default=2, help="Min word frequency for vocab")
    parser.add_argument("--max_length", type=int, default=20, help="Max sentence length (words)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()
    train(args)
