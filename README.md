# Seq2Seq French-to-English Translator

A complete neural machine translation system using a **Sequence-to-Sequence LSTM model with Bahdanau Attention**, served through a modern web application.

![Architecture](https://img.shields.io/badge/Architecture-Encoder--Decoder_LSTM-blue)
![Attention](https://img.shields.io/badge/Mechanism-Bahdanau_Attention-cyan)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![Frontend](https://img.shields.io/badge/Frontend-Next.js_16-black)

## Overview

This project implements a French→English translator from scratch:
- **Encoder**: Bidirectional LSTM that processes French input sequences
- **Attention**: Bahdanau-style attention mechanism for alignment
- **Decoder**: LSTM decoder with context-aware generation
- **Frontend**: Beautiful Next.js web app with real-time translation

## Project Structure

```
Sequence2Sequence/
├── backend/
│   ├── model.py          # Encoder, Attention, Decoder, Seq2Seq model classes
│   ├── vocab.py          # Vocabulary building and text processing utilities
│   ├── train.py          # Training script (run to train the model)
│   ├── main.py           # FastAPI server with translation API
│   ├── requirements.txt  # Python dependencies
│   ├── data/             # Place fra.txt here
│   └── weights/          # Trained model weights saved here
├── frontend/
│   ├── app/              # Next.js app router pages
│   ├── components/       # UI components (SpotlightCard, GradientText, etc.)
│   ├── lib/              # Utilities
│   └── package.json      # Node dependencies
└── Copy_of_Seq2SeqModel.ipynb  # Original Colab notebook
```

## Setup & Usage

### 1. Download Training Data

Download the French-English parallel corpus from [Tatoeba/ManyThings.org](https://www.manythings.org/anki/fra-eng.zip):

```bash
# Download and extract
cd backend/data
curl -O https://www.manythings.org/anki/fra-eng.zip
unzip fra-eng.zip
# This creates fra.txt in the data/ directory
```

### 2. Train the Model

```bash
cd backend
pip install -r requirements.txt

# Train (adjust epochs/batch_size as needed)
python train.py --data data/fra.txt --epochs 10 --batch_size 64 --max_length 20
```

Training will save:
- `weights/model.pt` - Model weights
- `weights/fr_vocab.json` - French vocabulary
- `weights/en_vocab.json` - English vocabulary
- `weights/config.json` - Model configuration

### 3. Start the Backend Server

```bash
cd backend
python main.py
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:3000
```

### 5. Open in Browser

Navigate to `http://localhost:3000` and start translating!

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/translate` | Translate French text to English |
| GET | `/api/health` | Check if model is loaded |
| GET | `/api/examples` | Get example sentences |

### Example Request

```bash
curl -X POST http://localhost:8000/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?"}'
```

## Model Architecture

```
┌─────────────────────┐
│   French Input      │  "Bonjour le monde"
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Embedding (256d)   │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Encoder LSTM (512) │ → Hidden State + All Outputs
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Attention (512→1)  │ → Context Vector (weighted sum)
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Decoder LSTM (768→512) │ → Predictions
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│  Linear (512→vocab) │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│   English Output    │  "Hello the world"
└─────────────────────┘
```

## Tech Stack

- **Model**: PyTorch (LSTM + Bahdanau Attention)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Next.js 16 + Tailwind CSS + Framer Motion
- **UI Components**: SpotlightCard, GradientText, BlurText (from NxtDevs)

## Training Details

- **Dataset**: Tatoeba French-English parallel corpus (~217K pairs)
- **Embedding Dim**: 256
- **Hidden Dim**: 512
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss (ignoring padding)
- **Gradient Clipping**: max_norm=1.0
- **Teacher Forcing**: 100% during training

## License

MIT
