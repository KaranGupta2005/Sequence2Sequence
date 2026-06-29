"""
Training script for the Seq2Seq French-to-English translation model.
Run this script to train the model and save the weights + vocabulary.

Usage:
    python train.py --data fra.txt --epochs 10 --batch_size 64
"""

import argparse
import os
import json
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
    src_pad = pad_sequence(src, batch_first=True, padding_value=0)
    trg_pad = pad_sequence(trg, batch_first=True, padding_value=0)
    return src_pad, trg_pad


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading data...")
    pairs = load_pairs(args.data)
    print(f"Loaded {len(pairs)} sentence pairs")

    # Filter to shorter sentences for faster training and better results
    if args.max_length:
        pairs = [(fr, en) for fr, en in pairs if len(fr.split()) <= args.max_length and len(en.split()) <= args.max_length]
        print(f"Filtered to {len(pairs)} pairs with max length {args.max_length}")

    # Build vocabularies
    fr_sentences = [p[0] for p in pairs]
    en_sentences = [p[1] for p in pairs]

    fr_vocab = build_vocab(fr_sentences, min_freq=args.min_freq)
    en_vocab = build_vocab(en_sentences, min_freq=args.min_freq)

    print(f"French vocab size: {len(fr_vocab)}")
    print(f"English vocab size: {len(en_vocab)}")

    # Save vocabularies
    os.makedirs(args.output_dir, exist_ok=True)
    save_vocab(fr_vocab, os.path.join(args.output_dir, "fr_vocab.json"))
    save_vocab(en_vocab, os.path.join(args.output_dir, "en_vocab.json"))

    # Save model config
    config = {
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "fr_vocab_size": len(fr_vocab),
        "en_vocab_size": len(en_vocab),
        "max_length": args.max_length or 50,
    }
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Create data loader
    dataset = TranslationDataset(pairs, fr_vocab, en_vocab)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create model
    encoder = Encoder(len(fr_vocab), args.embed_dim, args.hidden_dim)
    attention = Attention(args.hidden_dim)
    decoder = Decoder(len(en_vocab), args.embed_dim, args.hidden_dim, attention)
    model = Seq2Seq(encoder, decoder, len(en_vocab)).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Starting training for {args.epochs} epochs...")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)

            outputs = model(src, trg)
            outputs = outputs[:, 1:].reshape(-1, outputs.shape[-1])
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Seq2Seq French-to-English Translation Model")
    parser.add_argument("--data", type=str, default="data/fra.txt", help="Path to fra.txt data file")
    parser.add_argument("--output_dir", type=str, default="weights", help="Directory to save model and vocab")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum word frequency for vocab")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum sentence length (words)")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    train(args)
