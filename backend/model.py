"""
Seq2Seq Model with Attention for French to English Translation.
This module defines the Encoder, Attention, Decoder, and Seq2Seq model classes.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim: int, embed_dim: int, hidden_dim: int, attention: Attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = attention
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, output_dim: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_dim = output_dim

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        outputs = torch.zeros(batch_size, trg_len, self.output_dim)

        input = trg[:, 0]
        encoder_outputs, (hidden, cell) = self.encoder(src)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            input = trg[:, t]  # teacher forcing

        return outputs
