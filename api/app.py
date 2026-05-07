import os
import torch
from contextlib import asynccontextmanager
from typing import Optional

from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ─── Config ───────────────────────────────────────────────────────────────────
ADAPTER            = os.getenv("ADAPTER", "robsonrtp/ngananlp-v2")
BASE_MODEL         = "facebook/nllb-200-distilled-1.3B"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN           = os.getenv("HF_TOKEN", None)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", None)

# Default voice: "Rachel" (premade, available on free tier)
DEFAULT_VOICE_ID   = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

LANGUAGES = {
    "por_Latn": "Português",
    "lin_Latn": "Lingala",
    "umb_Latn": "Umbundu",
    "kmb_Latn": "Kimbundu",
    "cjk_Latn": "Tchokwe",
    "lue_Latn": "Luvale",
}

# Direcções com baixa confiança (X → lue_Latn)
LOW_CONFIDENCE_TARGETS = {"lue_Latn"}

# ─── Model loading ────────────────────────────────────────────────────────────
tokenizer = None
model     = None

def load_model():
    global tokenizer, model
    print(f"Loading tokenizer from {ADAPTER}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, token=HF_TOKEN)

    print(f"Loading base model {BASE_MODEL}...", flush=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    base = base.to(DEVICE)

    print(f"Loading LoRA adapter from {ADAPTER}...", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER, token=HF_TOKEN).eval()
    print("Model ready.", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NganaNLP v2",
    description="Multilingual translation API for Portuguese and Angolan Bantu languages.\n\nExample: POST /translate `{\"text\": \"Como tomar boas decisões sobre educação?\", \"src_lang\": \"por_Latn\", \"tgt_lang\": \"kmb_Latn\"}`",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str
    num_beams: Optional[int] = 4

    @field_validator("src_lang", "tgt_lang")
    @classmethod
    def validate_lang(cls, v):
        if v not in LANGUAGES:
            raise ValueError(f"Unsupported language '{v}'. Use one of: {list(LANGUAGES)}")
        return v

    @field_validator("num_beams")
    @classmethod
    def validate_beams(cls, v):
        if not (1 <= v <= 8):
            raise ValueError("num_beams must be between 1 and 8")
        return v

class SpeakRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None  # overrides DEFAULT_VOICE_ID if provided

class TranslateResponse(BaseModel):
    translation: str
    src_lang: str
    tgt_lang: str
    low_confidence: bool

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "NganaNLP v2",
        "description": "Multilingual translation API for Portuguese and Angolan Bantu languages",
        "message": "A language unheard is a thought untranslated. We build bridges.",
        "languages": LANGUAGES,
        "endpoints": {
            "translate": "POST /translate",
            "speak": "POST /speak",
            "languages": "GET /languages",
            "docs": "GET /docs",
            "health": "GET /health",
        },
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": ADAPTER}

@app.get("/languages")
def languages():
    return {
        "languages": [
            {"code": code, "name": name, "unreliable_as_target": code in LOW_CONFIDENCE_TARGETS}
            for code, name in LANGUAGES.items()
        ]
    }

@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if req.src_lang == req.tgt_lang:
        raise HTTPException(status_code=400, detail="src_lang and tgt_lang must be different")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    tokenizer.src_lang = req.src_lang
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=192,
    ).to(DEVICE)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(req.tgt_lang)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=req.num_beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    translation = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return TranslateResponse(
        translation=translation,
        src_lang=req.src_lang,
        tgt_lang=req.tgt_lang,
        low_confidence=req.tgt_lang in LOW_CONFIDENCE_TARGETS,
    )


# ─── TTS ──────────────────────────────────────────────────────────────────────
@app.post(
    "/speak",
    summary="Text-to-speech via ElevenLabs",
    responses={200: {"content": {"audio/mpeg": {}}}},
)
def speak(req: SpeakRequest):
    """Convert text to speech and return a complete MP3 audio file."""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    voice_id = req.voice_id or DEFAULT_VOICE_ID

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    audio_chunks = client.text_to_speech.convert(
        text=req.text,
        voice_id=voice_id,
        model_id="eleven_v3",
        output_format="mp3_44100_128",
    )

    audio_bytes = b"".join(audio_chunks)

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )
