from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
from pathlib import Path
from .pipeline import HybridPipeline

app = FastAPI(title="Text Classification API", version="1.0.0")
pipeline = HybridPipeline()

DATA_PATH = Path(__file__).parent.parent / "data" / "sample_logs.csv"


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    label: str
    method: str
    confidence: float | None = None


class TrainRequest(BaseModel):
    csv_path: str | None = None


@app.get("/")
def root():
    return {"message": "Text Classification API is running"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    try:
        result = pipeline.classify(req.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
    return ClassifyResponse(**result)


@app.post("/train")
def train(req: TrainRequest | None = None):
    csv_path = Path(req.csv_path) if req and req.csv_path else DATA_PATH
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "log_message" not in df.columns or "label" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must have 'log_message' and 'label' columns")
    pipeline.train_bert(df["log_message"].tolist(), df["label"].tolist())
    return {"message": "BERT model trained successfully", "samples": len(df)}


@app.post("/train/upload")
async def train_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are supported.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")
    if "log_message" not in df.columns or "label" not in df.columns:
        raise HTTPException(status_code=422, detail="CSV must have 'log_message' and 'label' columns")
    df = df.dropna(subset=["log_message", "label"])
    pipeline.train_bert(df["log_message"].tolist(), df["label"].tolist())
    counts = df["label"].value_counts().to_dict()
    return {"message": "BERT model trained successfully", "samples": len(df), "class_counts": counts}


@app.get("/stats")
def stats():
    return pipeline.training_stats()


@app.post("/classify/batch")
def classify_batch(texts: list[str]):
    if not texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")
    try:
        results = [pipeline.classify(t) for t in texts]
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
    return results
