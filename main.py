import httpx
from llama_index.core import SimpleDirectoryReader

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_CONFIG = {"model": "starling-lm:7b-alpha-q5_K_M", "stream": False}

docs = SimpleDirectoryReader("./data").load_data()

prompt = f"Extract address of the STOP 1 {docs[1].text}"


response = httpx.post(
    OLLAMA_ENDPOINT,
    json={"prompt": prompt, **OLLAMA_CONFIG},
    headers={"Content-Type": "application/json"},
    timeout=None,
)

print(response.json()["response"].strip())
