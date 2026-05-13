FROM python:3.11.9-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only backend code and data
COPY backend/ ./backend/
COPY data/ ./data/

# Azure App Service uses PORT env variable
ENV PORT=8000
ENV GROQ_API_KEY=""
ENV LLM_MODEL="deepseek-r1-distill-llama-70b"
ENV MIN_SAMPLES_FOR_BERT="10"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
