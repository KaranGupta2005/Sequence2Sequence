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

from model import Encoder, Attention, Decoder, Seq2Seq
from vocab import clean_text, tokenize, encode, load_vocab

app = FastAPI(
    title="Seq2Seq French-to-English Translator",
    description="Neural Machine Translation using LSTM with Attention",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and vocab
model = None
fr_vocab = None
en_vocab = None
inv_en_vocab = None
config = None
device = torch.device('cpu')


class TranslateRequest(BaseModel):
    text: str
    max_length: Optional[int] = 50


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
    inv_en_vocab = {v: k for k, v in en_vocab.items()}

    # Create model
    encoder = Encoder(config["fr_vocab_size"], config["embed_dim"], config["hidden_dim"])
    attention = Attention(config["hidden_dim"])
    decoder = Decoder(config["en_vocab_size"], config["embed_dim"], config["hidden_dim"], attention)
    model = Seq2Seq(encoder, decoder, config["en_vocab_size"]).to(device)

    # Load weights
    model_path = os.path.join(weights_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"WARNING: Model weights not found at {model_path}. Model not loaded.")
        return False

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  French vocab: {len(fr_vocab)} tokens")
    print(f"  English vocab: {len(en_vocab)} tokens")
    print(f"  Device: {device}")

    return True


def translate_sentence(sentence: str, max_length: int = 50) -> tuple:
    """Translate a French sentence to English."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    cleaned = clean_text(sentence)
    src = torch.tensor(encode(cleaned, fr_vocab), dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src)

        input_token = torch.tensor([en_vocab["<start>"]], dtype=torch.long).to(device)
        outputs = []
        attention_weights = []

        for _ in range(max_length):
            # Get attention weights
            attn_w = model.decoder.attention(hidden, encoder_outputs)
            attention_weights.append(attn_w.squeeze(0).cpu().tolist())

            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            pred = output.argmax(1).item()

            if pred == en_vocab["<end>"]:
                break

            outputs.append(pred)
            input_token = torch.tensor([pred], dtype=torch.long).to(device)

    translated = " ".join([inv_en_vocab.get(i, "<unk>") for i in outputs])
    return translated, cleaned, attention_weights


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = load_model()
    if not success:
        print("Model could not be loaded. Train the model first using train.py")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check if the model is loaded and ready."""
    return HealthResponse(
        status="ready" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        fr_vocab_size=len(fr_vocab) if fr_vocab else 0,
        en_vocab_size=len(en_vocab) if en_vocab else 0,
    )


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Translate French text to English."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running: python train.py --data data/fra.txt"
        )

    try:
        translated, cleaned, attn_weights = translate_sentence(
            request.text,
            max_length=request.max_length or 50
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
