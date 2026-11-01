
from pathlib import Path
from contextlib import asynccontextmanager

import re
import pickle
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = BASE_DIR / "Artifacts"
STATIC_DIR = BASE_DIR / "static"

model_path = ARTIFACTS_DIR / "BiGRU_Model.keras"
tokenizer_path = ARTIFACTS_DIR / "tokenizer.pkl"

max_sequence_length = 50

emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲"
}


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]
    emoji: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading the model and tokenizer...")

    try:
        if not model_path.exists():
            raise FileNotFoundError(
                f"BiGRU model not found: {model_path}"
            )

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}"
            )

        dl_model["BiGRU"] = load_model(str(model_path))

        with open(tokenizer_path, "rb") as file:
            dl_model["Tokenizer"] = pickle.load(file)

        print("Model and tokenizer loaded successfully!")

    except Exception as e:
        dl_model.clear()
        print(f"Model loading failed: {e}")
        raise

    yield

    dl_model.clear()


app = FastAPI(
    title="Emotion Prediction API",
    description="Emotion prediction using BiGRU deep learning model.",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )


@app.get("/", include_in_schema=False)
def server_ui():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(str(index_file))


@app.get("/health", response_model=HealthResponse)
def health_check():

    model_loaded = (
        "BiGRU" in dl_model
        and "Tokenizer" in dl_model
    )

    return HealthResponse(
        status="Server is running",
        model_loaded=model_loaded
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer_model = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again later."
        )

    cleaned_text = preprocess_text(text_input.text)

    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Please enter valid text."
        )

    try:
        tokenized_text = tokenizer_model.texts_to_sequences(
            [cleaned_text]
        )

        padded_sequence = pad_sequences(
            tokenized_text,
            maxlen=max_sequence_length,
            padding="post",
            truncating="post"
        )

        probabilities = BiGRU_model.predict(
            padded_sequence,
            verbose=0
        )[0]

        if len(probabilities) != len(emotion_labels):
            raise ValueError(
                "Model output does not match emotion labels."
            )

        top_emotion_index = int(np.argmax(probabilities))

        predicted_emotion = emotion_labels[top_emotion_index]

        all_probabilities = {
            label: float(prob)
            for label, prob in zip(
                emotion_labels,
                probabilities
            )
        }

        return PredictionResponse(
            text=text_input.text,
            predicted_emotion=predicted_emotion,
            confidence=float(probabilities[top_emotion_index]),
            all_probabilities=all_probabilities,
            emoji=EMOTION_EMOJIS.get(predicted_emotion, "")
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
