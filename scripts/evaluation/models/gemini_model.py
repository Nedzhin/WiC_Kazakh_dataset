import os
import google.generativeai as genai

class GeminiChat:
    """Gemini 1.5 Pro/Flash via google-generativeai."""
    def __init__(self, model: str = "gemini-1.5-pro", api_key_env: str = "GOOGLE_API_KEY"):
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)

    def infer(self, prompt: str) -> str:
        out = self.model.generate_content(prompt)
        return getattr(out, "text", "").strip()
