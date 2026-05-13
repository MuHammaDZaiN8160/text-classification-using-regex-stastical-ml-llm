import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "lr_model.joblib"
LABELS_PATH = MODELS_DIR / "train_labels.npy"


class BertClassifier:
    """TF-IDF + Logistic Regression classifier (lightweight, no torch needed)."""

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.encoder: LabelEncoder | None = None
        self._load_if_exists()

    def _load_if_exists(self):
        if MODEL_PATH.exists():
            saved = joblib.load(MODEL_PATH)
            self.pipeline = saved["pipeline"]
            self.encoder  = saved["encoder"]

    def is_trained(self) -> bool:
        return self.pipeline is not None

    def train(self, texts: list[str], labels: list[str]):
        MODELS_DIR.mkdir(exist_ok=True)
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(labels)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ("clf",   LogisticRegression(max_iter=500, C=5.0)),
        ])
        self.pipeline.fit(texts, y)
        joblib.dump({"pipeline": self.pipeline, "encoder": self.encoder}, MODEL_PATH)
        np.save(LABELS_PATH, np.array(labels))

    def predict(self, text: str) -> str:
        if not self.is_trained():
            raise RuntimeError("Classifier not trained yet.")
        idx = self.pipeline.predict([text])[0]
        return self.encoder.inverse_transform([idx])[0]

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        if not self.is_trained():
            raise RuntimeError("Classifier not trained yet.")
        proba      = self.pipeline.predict_proba([text])[0]
        idx        = proba.argmax()
        confidence = float(round(proba[idx], 4))
        label      = self.encoder.inverse_transform([idx])[0]
        return label, confidence

    def class_counts(self) -> dict[str, int]:
        if not LABELS_PATH.exists():
            return {}
        labels = np.load(LABELS_PATH, allow_pickle=True)
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
