"""
Improved Seq2Seq Model with Attention for French to English Translation.
Bidirectional Encoder, 2-Layer LSTM, Dropout, Beam Search.
"""

import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)))
        cell = cell.view(self.n_layers, 2, -1, self.hidden_dim)
        cell = torch.tanh(self.fc_cell(torch.cat([cell[:, 0], cell[:, 1]], dim=2)))
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_dim, encoder_dim=None):
        super().__init__()
        encoder_dim = encoder_dim or hidden_dim * 2
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        h = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention, n_layers=2, dropout=0.3):
        super().__init__()
        encoder_dim = hidden_dim * 2
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(embed_dim + encoder_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        output, (hidden, cell) = self.lstm(torch.cat((embedded, context), dim=2), (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_dim, device=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_dim = output_dim
        self.device = device or torch.device('cpu')

    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        batch_size, trg_len = trg.shape
        outputs = torch.zeros(batch_size, trg_len, self.output_dim).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            input = trg[:, t] if random.random() < teacher_forcing_ratio else output.argmax(1)
        return outputs
