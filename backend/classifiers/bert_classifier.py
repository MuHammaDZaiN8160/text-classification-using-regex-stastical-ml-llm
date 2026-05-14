import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

MODELS_DIR  = Path(__file__).parent.parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "lr_model.joblib"
LABELS_PATH = MODELS_DIR / "train_labels.npy"

BERT_MODEL = "all-MiniLM-L6-v2"  # 80MB, fast, BERT-based


class BertClassifier:
    def __init__(self):
        self.encoder: LabelEncoder | None  = None
        self.lr: LogisticRegression | None = None
        self._embedder: SentenceTransformer | None = None
        self._load_if_exists()

    def _load_if_exists(self):
        if MODEL_PATH.exists():
            saved        = joblib.load(MODEL_PATH)
            self.lr      = saved["lr"]
            self.encoder = saved["encoder"]

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(BERT_MODEL)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._get_embedder().encode(texts, show_progress_bar=False)

    def is_trained(self) -> bool:
        return self.lr is not None

    def train(self, texts: list[str], labels: list[str]):
        MODELS_DIR.mkdir(exist_ok=True)
        embeddings   = self._embed(texts)
        self.encoder = LabelEncoder()
        y            = self.encoder.fit_transform(labels)
        self.lr      = LogisticRegression(max_iter=500, C=5.0)
        self.lr.fit(embeddings, y)
        joblib.dump({"lr": self.lr, "encoder": self.encoder}, MODEL_PATH)
        np.save(LABELS_PATH, np.array(labels))

    def predict(self, text: str) -> str:
        if not self.is_trained():
            raise RuntimeError("BertClassifier not trained yet.")
        emb = self._embed([text])
        idx = self.lr.predict(emb)[0]
        return self.encoder.inverse_transform([idx])[0]

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        if not self.is_trained():
            raise RuntimeError("BertClassifier not trained yet.")
        emb        = self._embed([text])
        proba      = self.lr.predict_proba(emb)[0]
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
