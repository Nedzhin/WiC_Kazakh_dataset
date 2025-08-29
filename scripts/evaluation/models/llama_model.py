import requests

class OllamaChat:
    """
    Llama-3 via local Ollama REST API.
    You should do `ollama serve` and `ollama pull llama3` beforehand.
    """
    def __init__(self, model: str = "llama3",
                 url: str = "http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    def infer(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 8,   # enough for True/False
                "num_ctx": 1024      # keeping memory small for 8 GB MacOS
            }
        }
        r = requests.post(self.url, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()
