"""
FastAPI backend for the Seq2Seq French-to-English Translation model.
Supports both old (single-layer) and new (bidirectional 2-layer) architectures.
"""

import os
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

from model import Encoder, Attention, Decoder, Seq2Seq
from vocab import clean_text, encode, load_vocab

# Global
model = None
fr_vocab = None
en_vocab = None
inv_en_vocab = None
config = None
device = torch.device('cpu')


def load_model():
    global model, fr_vocab, en_vocab, inv_en_vocab, config, device

    weights_dir = os.environ.get("WEIGHTS_DIR", "weights")

    config_path = os.path.join(weights_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"WARNING: Config not found at {config_path}.")
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    fr_vocab_path = os.path.join(weights_dir, "fr_vocab.json")
    en_vocab_path = os.path.join(weights_dir, "en_vocab.json")

    if not os.path.exists(fr_vocab_path) or not os.path.exists(en_vocab_path):
        print("WARNING: Vocabulary files not found.")
        return False

    fr_vocab = load_vocab(fr_vocab_path)
    en_vocab = load_vocab(en_vocab_path)
    inv_en_vocab = {int(v): k for k, v in en_vocab.items()}

    # Read architecture params from config
    n_layers = config.get("n_layers", 2)
    dropout = config.get("dropout", 0.3)

    encoder = Encoder(config["fr_vocab_size"], config["embed_dim"], config["hidden_dim"],
                      n_layers=n_layers, dropout=dropout)
    attention = Attention(config["hidden_dim"])
    decoder = Decoder(config["en_vocab_size"], config["embed_dim"], config["hidden_dim"], attention,
                      n_layers=n_layers, dropout=dropout)
    model = Seq2Seq(encoder, decoder, config["en_vocab_size"], device).to(device)

    model_path = os.path.join(weights_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"WARNING: Model weights not found at {model_path}.")
        return False

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Architecture: Bi-LSTM {n_layers}L, {config['embed_dim']}emb, {config['hidden_dim']}hid")
    print(f"  French vocab: {len(fr_vocab)} tokens")
    print(f"  English vocab: {len(en_vocab)} tokens")
    print(f"  Device: {device}")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    success = load_model()
    if not success:
        print("Model could not be loaded. Train the model first.")
    yield


app = FastAPI(
    title="Seq2Seq French-to-English Translator",
    description="Neural Machine Translation using Bi-LSTM with Attention",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str
    max_length: Optional[int] = 30


class TranslateResponse(BaseModel):
    input_text: str
    translated_text: str
    cleaned_input: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    fr_vocab_size: int
    en_vocab_size: int


def translate_sentence(sentence: str, max_length: int = 30) -> tuple:
    """Translate using greedy decoding."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cleaned = clean_text(sentence)
    src = torch.tensor(encode(cleaned, fr_vocab), dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src)
        input_token = torch.tensor([en_vocab["<start>"]], dtype=torch.long).to(device)
        outputs = []

        for _ in range(max_length):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            pred = output.argmax(1).item()
            if pred == en_vocab["<end>"]:
                break
            outputs.append(pred)
            input_token = torch.tensor([pred], dtype=torch.long).to(device)

    translated = " ".join([inv_en_vocab.get(i, "<unk>") for i in outputs])

    # Post-processing: remove consecutive duplicate words
    words = translated.split()
    cleaned_words = []
    for i, w in enumerate(words):
        if i == 0 or w != words[i - 1]:
            cleaned_words.append(w)
    translated = " ".join(cleaned_words)

    return translated, cleaned


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ready" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        fr_vocab_size=len(fr_vocab) if fr_vocab else 0,
        en_vocab_size=len(en_vocab) if en_vocab else 0,
    )


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        translated, cleaned = translate_sentence(request.text, request.max_length or 30)
        return TranslateResponse(
            input_text=request.text,
            translated_text=translated,
            cleaned_input=cleaned,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.get("/api/examples")
async def get_examples():
    examples = [
        {"french": "Bonjour, comment allez-vous?", "expected": "Hello, how are you?"},
        {"french": "Je suis un étudiant.", "expected": "I am a student."},
        {"french": "Merci beaucoup.", "expected": "Thank you very much."},
        {"french": "Je t'aime.", "expected": "I love you."},
        {"french": "Bonne nuit.", "expected": "Good night."},
        {"french": "Je ne comprends pas.", "expected": "I do not understand."},
        {"french": "Il fait froid.", "expected": "It is cold."},
        {"french": "Où est la gare?", "expected": "Where is the station?"},
    ]
    return {"examples": examples}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
