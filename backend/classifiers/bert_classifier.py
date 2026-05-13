import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "lr_model.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
EMBEDDINGS_PATH = MODELS_DIR / "train_embeddings.npy"
LABELS_PATH = MODELS_DIR / "train_labels.npy"


def _get_embedder():
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _embed(texts: list[str], tokenizer, model) -> np.ndarray:
    import torch

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()


class BertClassifier:
    def __init__(self):
        self.lr: LogisticRegression | None = None
        self.encoder: LabelEncoder | None = None
        self._tokenizer = None
        self._model = None
        self._load_if_exists()

    def _load_if_exists(self):
        if MODEL_PATH.exists() and ENCODER_PATH.exists():
            self.lr = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)

    def _ensure_embedder(self):
        if self._tokenizer is None:
            self._tokenizer, self._model = _get_embedder()

    def is_trained(self) -> bool:
        return self.lr is not None

    def train(self, texts: list[str], labels: list[str]):
        MODELS_DIR.mkdir(exist_ok=True)
        self._ensure_embedder()
        embeddings = _embed(texts, self._tokenizer, self._model)
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(labels)
        self.lr = LogisticRegression(max_iter=500)
        self.lr.fit(embeddings, y)
        joblib.dump(self.lr, MODEL_PATH)
        joblib.dump(self.encoder, ENCODER_PATH)
        np.save(EMBEDDINGS_PATH, embeddings)
        np.save(LABELS_PATH, np.array(labels))

    def predict(self, text: str) -> str:
        if not self.is_trained():
            raise RuntimeError("BertClassifier not trained yet.")
        self._ensure_embedder()
        emb = _embed([text], self._tokenizer, self._model)
        idx = self.lr.predict(emb)[0]
        return self.encoder.inverse_transform([idx])[0]

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        if not self.is_trained():
            raise RuntimeError("BertClassifier not trained yet.")
        self._ensure_embedder()
        emb = _embed([text], self._tokenizer, self._model)
        proba = self.lr.predict_proba(emb)[0]
        idx = proba.argmax()
        confidence = float(round(proba[idx], 4))
        label = self.encoder.inverse_transform([idx])[0]
        return label, confidence

    def class_counts(self) -> dict[str, int]:
        if not LABELS_PATH.exists():
            return {}
        labels = np.load(LABELS_PATH, allow_pickle=True)
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
