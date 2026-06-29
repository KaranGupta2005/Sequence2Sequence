"""
FastAPI backend for the Seq2Seq French-to-English Translation model.
Serves the trained model via REST API endpoints.
"""

import os
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

from model import Encoder, Attention, Decoder, Seq2Seq
from vocab import clean_text, tokenize, encode, load_vocab


# Global model and vocab
model = None
fr_vocab = None
en_vocab = None
inv_en_vocab = None
config = None
device = torch.device('cpu')


def load_model():
    """Load the trained model and vocabularies."""
    global model, fr_vocab, en_vocab, inv_en_vocab, config, device

    weights_dir = os.environ.get("WEIGHTS_DIR", "weights")

    # Load config
    config_path = os.path.join(weights_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"WARNING: Config not found at {config_path}. Model not loaded.")
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load vocabularies
    fr_vocab_path = os.path.join(weights_dir, "fr_vocab.json")
    en_vocab_path = os.path.join(weights_dir, "en_vocab.json")

    if not os.path.exists(fr_vocab_path) or not os.path.exists(en_vocab_path):
        print("WARNING: Vocabulary files not found. Model not loaded.")
        return False

    fr_vocab = load_vocab(fr_vocab_path)
    en_vocab = load_vocab(en_vocab_path)
    inv_en_vocab = {int(v): k for k, v in en_vocab.items()}

    # Create model with improved architecture
    n_layers = config.get("n_layers", 2)
    dropout = config.get("dropout", 0.3)

    encoder = Encoder(config["fr_vocab_size"], config["embed_dim"], config["hidden_dim"],
                      n_layers=n_layers, dropout=dropout)
    attention = Attention(config["hidden_dim"])
    decoder = Decoder(config["en_vocab_size"], config["embed_dim"], config["hidden_dim"], attention,
                      n_layers=n_layers, dropout=dropout)
    model = Seq2Seq(encoder, decoder, config["en_vocab_size"], device).to(device)

    # Load weights
    model_path = os.path.join(weights_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"WARNING: Model weights not found at {model_path}. Model not loaded.")
        return False

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  French vocab: {len(fr_vocab)} tokens")
    print(f"  English vocab: {len(en_vocab)} tokens")
    print(f"  Architecture: Bi-LSTM {n_layers}L, {config['embed_dim']}emb, {config['hidden_dim']}hid")
    print(f"  Device: {device}")

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    success = load_model()
    if not success:
        print("Model could not be loaded. Train the model first using train.py")
    yield


app = FastAPI(
    title="Seq2Seq French-to-English Translator",
    description="Neural Machine Translation using Bi-LSTM with Attention & Beam Search",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str
    max_length: Optional[int] = 50
    beam_width: Optional[int] = 5


class TranslateResponse(BaseModel):
    input_text: str
    translated_text: str
    cleaned_input: str
    attention_weights: Optional[List[List[float]]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    fr_vocab_size: int
    en_vocab_size: int
    architecture: str


def translate_sentence(sentence: str, max_length: int = 50, beam_width: int = 5) -> tuple:
    """Translate a French sentence to English using beam search."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cleaned = clean_text(sentence)
    src = torch.tensor(encode(cleaned, fr_vocab), dtype=torch.long).unsqueeze(0).to(device)

    # Use beam search for better quality
    tokens = model.beam_search(src, en_vocab["<start>"], en_vocab["<end>"], max_length, beam_width)

    # Convert to words
    words = []
    for t in tokens:
        if t == en_vocab.get("<end>", -1):
            break
        word = inv_en_vocab.get(t, "<unk>")
        if word not in ("<start>", "<end>", "<pad>", "<unk>"):
            words.append(word)

    translated = " ".join(words)

    # Also get attention weights (greedy pass for visualization)
    attention_weights = []
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src)
        input_token = torch.tensor([en_vocab["<start>"]], dtype=torch.long).to(device)
        for _ in range(min(max_length, 30)):
            attn_w = model.decoder.attention(hidden, encoder_outputs)
            attention_weights.append(attn_w.squeeze(0).cpu().tolist())
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            pred = output.argmax(1).item()
            if pred == en_vocab["<end>"]:
                break
            input_token = torch.tensor([pred], dtype=torch.long).to(device)

    return translated, cleaned, attention_weights


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check if the model is loaded and ready."""
    arch = "unknown"
    if config:
        arch = f"Bi-LSTM {config.get('n_layers', 2)}L, {config.get('embed_dim', 256)}emb, {config.get('hidden_dim', 512)}hid"
    return HealthResponse(
        status="ready" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        fr_vocab_size=len(fr_vocab) if fr_vocab else 0,
        en_vocab_size=len(en_vocab) if en_vocab else 0,
        architecture=arch,
    )


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Translate French text to English."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first: python train.py --data data/fra.txt"
        )

    try:
        translated, cleaned, attn_weights = translate_sentence(
            request.text,
            max_length=request.max_length or 50,
            beam_width=request.beam_width or 5
        )

        return TranslateResponse(
            input_text=request.text,
            translated_text=translated,
            cleaned_input=cleaned,
            attention_weights=attn_weights
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.get("/api/examples")
async def get_examples():
    """Return example French sentences for testing."""
    examples = [
        {"french": "Bonjour, comment allez-vous?", "expected": "Hello, how are you?"},
        {"french": "Je suis un étudiant.", "expected": "I am a student."},
        {"french": "Il fait beau aujourd'hui.", "expected": "It is nice today."},
        {"french": "Merci beaucoup.", "expected": "Thank you very much."},
        {"french": "Je t'aime.", "expected": "I love you."},
        {"french": "Où est la gare?", "expected": "Where is the station?"},
        {"french": "J'ai faim.", "expected": "I am hungry."},
        {"french": "Bonne nuit.", "expected": "Good night."},
        {"french": "C'est très bien.", "expected": "It is very good."},
        {"french": "Je ne comprends pas.", "expected": "I do not understand."},
    ]
    return {"examples": examples}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
