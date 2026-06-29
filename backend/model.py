"""
Improved Seq2Seq Model with Attention for French to English Translation.

Improvements over baseline:
- Bidirectional Encoder (captures left and right context)
- 2-layer LSTM (deeper representations)
- Dropout regularization (prevents overfitting)
- Beam Search decoding (better output quality)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        # Project bidirectional hidden states to decoder size
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)

        # hidden: [n_layers * 2, batch, hidden_dim] -> [n_layers, batch, hidden_dim]
        # Concatenate forward and backward for each layer
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # [n_layers, batch, hidden_dim*2]
        hidden = torch.tanh(self.fc_hidden(hidden))  # [n_layers, batch, hidden_dim]

        cell = cell.view(self.n_layers, 2, -1, self.hidden_dim)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        cell = torch.tanh(self.fc_cell(cell))

        # outputs: [batch, seq_len, hidden_dim * 2] (bidirectional)
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, encoder_dim: int = None):
        super().__init__()
        encoder_dim = encoder_dim or hidden_dim * 2  # bidirectional
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [n_layers, batch, hidden_dim] -> use top layer
        hidden_top = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)

        energy = torch.tanh(self.attn(torch.cat((hidden_top, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim: int, embed_dim: int, hidden_dim: int, attention: Attention,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        encoder_dim = hidden_dim * 2  # bidirectional encoder output

        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(
            embed_dim + encoder_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        # Output projection with context
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Combine output, context, and embedded for prediction
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, output_dim: int, device=None):
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

            # Scheduled sampling: use predicted token sometimes
            if teacher_forcing_ratio < 1.0:
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                input = trg[:, t] if use_teacher else output.argmax(1)
            else:
                input = trg[:, t]

        return outputs

    def beam_search(self, src, start_token, end_token, max_length=50, beam_width=5):
        """Beam search decoding for better translation quality."""
        self.eval()
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = self.encoder(src)

            # Each beam: (score, tokens, hidden, cell)
            beams = [(0.0, [start_token], hidden, cell)]
            completed = []

            for _ in range(max_length):
                new_beams = []

                for score, tokens, h, c in beams:
                    input_token = torch.tensor([tokens[-1]], dtype=torch.long).to(self.device)
                    output, new_h, new_c = self.decoder(input_token, h, c, encoder_outputs)

                    log_probs = torch.log_softmax(output, dim=1)
                    top_probs, top_indices = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        new_score = score + top_probs[0, i].item()
                        new_token = top_indices[0, i].item()
                        new_tokens = tokens + [new_token]

                        if new_token == end_token:
                            # Length normalize
                            normalized_score = new_score / len(new_tokens)
                            completed.append((normalized_score, new_tokens))
                        else:
                            new_beams.append((new_score, new_tokens, new_h, new_c))

                # Keep top beams
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_width]

                if not beams:
                    break

                # Early stopping if we have enough completed
                if len(completed) >= beam_width:
                    break

            # If no completed beams, use best incomplete
            if not completed:
                completed = [(score / len(tokens), tokens, None, None) for score, tokens, _, _ in beams]

            # Return best
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1][1:]  # Remove start token
