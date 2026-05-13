import os
from dotenv import load_dotenv
from .classifiers import RegexClassifier, BertClassifier, LLMClassifier

load_dotenv()

MIN_SAMPLES = int(os.getenv("MIN_SAMPLES_FOR_BERT", "10"))


class HybridPipeline:
    def __init__(self):
        self.regex = RegexClassifier()
        self.bert = BertClassifier()
        self._llm: LLMClassifier | None = None

    def _get_llm(self) -> LLMClassifier:
        if self._llm is None:
            self._llm = LLMClassifier()
        return self._llm

    def classify(self, text: str) -> dict:
        # Stage 1: Regex
        regex_result = self.regex.predict(text)
        if regex_result:
            return {"label": regex_result, "method": "regex", "confidence": 1.0}

        # Stage 2: BERT + Logistic Regression (if trained with enough samples)
        counts = self.bert.class_counts()
        has_enough = self.bert.is_trained() and all(c >= MIN_SAMPLES for c in counts.values())
        if has_enough:
            label, confidence = self.bert.predict_with_confidence(text)
            return {"label": label, "method": "bert", "confidence": confidence}

        # Stage 3: LLM fallback
        label = self._get_llm().predict(text)
        return {"label": label, "method": "llm", "confidence": None}

    def train_bert(self, texts: list[str], labels: list[str]):
        self.bert.train(texts, labels)

    def training_stats(self) -> dict:
        return {
            "bert_trained": self.bert.is_trained(),
            "class_counts": self.bert.class_counts(),
            "min_samples_required": MIN_SAMPLES,
        }
