FROM python:3.11.9-slim

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install --no-cache-dir -r requirements-backend.txt

COPY backend/ ./backend/
COPY data/ ./data/
RUN mkdir -p models

ENV GROQ_API_KEY=""
ENV LLM_MODEL="deepseek-r1-distill-llama-70b"
ENV MIN_SAMPLES_FOR_BERT="10"

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
