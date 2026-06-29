# Seq2Seq French-to-English Translator

A complete neural machine translation system using a **Bidirectional Sequence-to-Sequence LSTM model with Bahdanau Attention**, served through a modern web application.

![Architecture](https://img.shields.io/badge/Architecture-Bi--LSTM_Encoder--Decoder-blue)
![Attention](https://img.shields.io/badge/Mechanism-Bahdanau_Attention-cyan)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Frontend](https://img.shields.io/badge/Frontend-Next.js_16-black)
![BLEU](https://img.shields.io/badge/BLEU--4-0.54-green)

---

## Model Performance

| Metric | Score |
|--------|-------|
| **BLEU-1** | 0.8187 |
| **BLEU-2** | 0.7290 |
| **BLEU-4** | 0.5412 |
| **Exact Match** | 38.7% |
| **Word-level Accuracy** | 70.5% |
| **Meaningful Translations (BLEU > 0.3)** | 65.0% |
| **High Quality (BLEU > 0.5)** | 57.3% |

> Evaluated on 300 randomly sampled sentence pairs (max 12 words each).

### Sample Translations

| French | Model Output | ✓ |
|--------|-------------|---|
| Merci beaucoup. | thank you very much. | ✅ |
| Je suis un homme. | i'm a man. | ✅ |
| Elle est belle. | she is beautiful. | ✅ |
| Je ne comprends pas. | i don't understand. | ✅ |
| Je t'aime. | i love you. | ✅ |
| Comment allez-vous? | how are you? | ✅ |
| Bonne nuit. | good night. | ✅ |
| Il fait froid. | it's cold. | ✅ |
| Je veux dormir. | i want to sleep. | ✅ |
| Bonjour, comment allez-vous? | hello, how are you? | ✅ |

---

## Architecture

```
┌─────────────────────────────┐
│   French Input              │  "je suis un homme ."
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Embedding (256 dim)        │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Bidirectional LSTM          │  2 layers, 512 hidden
│  (Encoder)                   │  Captures left + right context
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Bahdanau Attention          │  Aligns decoder focus to
│  (1024 → 512 → 1)           │  relevant encoder states
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  LSTM Decoder                │  2 layers, input = embed + context
│  (768 → 512)                 │  Generates one token at a time
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  FC Output (512+1024+256 → vocab) │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│   English Output             │  "i'm a man ."
└─────────────────────────────┘
```

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Total Parameters | 47,141,385 (~47M) |
| Encoder | Bidirectional LSTM, 2 layers |
| Decoder | LSTM, 2 layers |
| Embedding Dim | 256 |
| Hidden Dim | 512 |
| Dropout | 0.3 |
| French Vocab | 21,107 tokens |
| English Vocab | 12,041 tokens |
| Max Sequence Length | 20 words |

---

## Project Structure

```
Sequence2Sequence/
├── backend/
│   ├── model.py          # Bi-LSTM Encoder, Attention, Decoder, Seq2Seq
│   ├── vocab.py          # Tokenization & vocabulary utilities
│   ├── train.py          # Training script (scheduled sampling, val split)
│   ├── main.py           # FastAPI server with /api/translate endpoint
│   ├── requirements.txt  # Python dependencies
│   ├── Dockerfile        # Docker container for deployment
│   ├── data/             # Place fra.txt here
│   └── weights/          # Trained model weights (not in git)
├── frontend/
│   ├── app/              # Next.js app router
│   ├── components/       # UI components
│   └── package.json      # Node dependencies
├── docker-compose.yml    # Full stack deployment
├── DEPLOY.md             # Deployment guide (Firebase, Docker, Render)
└── run.bat               # Quick local start (Windows)
```

---

## Quick Start

### 1. Train the Model (or use pre-trained weights)

```bash
cd backend
pip install -r requirements.txt

# Download data
cd data
wget https://www.manythings.org/anki/fra-eng.zip
unzip fra-eng.zip
cd ..

# Train (GPU recommended, ~15 min on T4)
python train.py --data data/fra.txt --epochs 15 --batch_size 64
```

### 2. Start Backend

```bash
cd backend
python main.py
# Running at http://localhost:8000
```

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
# Running at http://localhost:3000
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/translate` | Translate French → English |
| GET | `/api/health` | Model status check |
| GET | `/api/examples` | Sample sentences |

### Example

```bash
curl -X POST http://localhost:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?"}'
```

Response:
```json
{
  "input_text": "Bonjour, comment allez-vous?",
  "translated_text": "hello , how are you ?",
  "cleaned_input": "bonjour , comment allez-vous ?"
}
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Dataset | Tatoeba French-English (~217K pairs) |
| Train/Val Split | 95% / 5% |
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| Loss | CrossEntropyLoss (ignore padding) |
| Gradient Clipping | max_norm=1.0 |
| Teacher Forcing | Scheduled (1.0 → 0.5 over epochs) |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Epochs | 15 |
| Batch Size | 64 |
| Training Time | ~3 hours (Kaggle CPU) |

---

## Deployment

See [DEPLOY.md](DEPLOY.md) for full deployment guide:
- **Firebase Hosting** + **Google Cloud Run**
- **Docker Compose**
- **Render / Railway**
- **Manual VPS**

---

## Tech Stack

- **Model**: PyTorch (Bi-LSTM + Bahdanau Attention)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Next.js 16 + Tailwind CSS + Framer Motion
- **UI**: SpotlightCard, GradientText, BlurText
- **Deployment**: Docker, Firebase, Cloud Run

---

## License

MIT
