# # test_openai.py
# from openai import OpenAI
# client = OpenAI()
# r = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role":"user","content":"Reply exactly: True"}],
#     temperature=0, max_tokens=3
# )
# print(r.choices[0].message.content)

# test_gemini.py
# import google.generativeai as genai, os
# from dotenv import load_dotenv
# ZERO_SHOT_KK = """Саған бірдей мақсатты сөзді қамтитын екі сөйлем берілген.
# Сенің тапсырмаң – осы сөздің екі сөйлемде де бірдей мағынада қолданылған-қолданылмағанын анықтау.
# Егер мағынасы бірдей болса, "True" деп жауап бер.
# Егер мағынасы әртүрлі болса, "False" деп жауап бер.
# Тек "True" немесе "False" деп жауап бер.

# Сөйлем 1: Бүгін күн жылы болады.
# Сөйлем 2: Біздің планета күнді айналады.
# Сөз: күн
# """
# load_dotenv()
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# m = genai.GenerativeModel("gemini-1.5-pro")
# out = m.generate_content(ZERO_SHOT_KK)
# print(getattr(out, "text", "").strip())


import requests
import json

def run_ollama_api(prompt: str, model: str = "llama3") -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post(url, json=payload, stream=True)

    # Collect streaming chunks
    output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                output += data["response"]
    return output.strip()

if __name__ == "__main__":
    prompt = """Саған бірдей мақсатты сөзді қамтитын екі сөйлем берілген.
# Сенің тапсырмаң – осы сөздің екі сөйлемде де бірдей мағынада қолданылған-қолданылмағанын анықтау.
# Егер мағынасы бірдей болса, "True" деп жауап бер.
# Егер мағынасы әртүрлі болса, "False" деп жауап бер.
# Тек "True" немесе "False" деп жауап бер.

# Сөйлем 1: Бүгін күн жылы болады.
# Сөйлем 2: Біздің планета күнді айналады.
# Сөз: күн
# """
    print(run_ollama_api(prompt, model="llama3"))

