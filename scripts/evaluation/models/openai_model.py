import os
from openai import OpenAI

class OpenAIChat:
    """GPT-4o via OpenAI SDK."""
    def __init__(self, model: str = "gpt-4o", api_key_env: str = "OPENAI_API_KEY"):
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set")
        self.client = OpenAI(api_key=key)
        self.model = model

    def infer(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=3
        )
        return r.choices[0].message.content.strip()
