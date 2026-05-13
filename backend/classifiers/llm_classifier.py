import os
from groq import Groq

LABELS = ["Security Alert", "Resource Usage", "Workflow Error"]

SYSTEM_PROMPT = """You are a log classification expert. Classify the given log message into exactly one of these categories:
- Security Alert: logs about unauthorized access, authentication failures, breaches, or suspicious activity
- Resource Usage: logs about memory, CPU, disk, network metrics or infrastructure capacity
- Workflow Error: logs about failed jobs, pipeline errors, task assignment failures, or process timeouts

Reply with ONLY the category name, nothing else."""


class LLMClassifier:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("LLM_MODEL", "deepseek-r1-distill-llama-70b")

    def predict(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Log message: {text}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip()
        for label in LABELS:
            if label.lower() in result.lower():
                return label
        return result
