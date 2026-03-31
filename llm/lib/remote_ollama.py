import os
import subprocess, time, requests

SSH_HOST   = os.getenv("SSH_HOST", None)
LOCAL_PORT = int(os.getenv("LOCAL_PORT", 18080))
REMOTE_PORT= int(os.getenv("REMOTE_PORT", 11434))
BASE_URL = f"http://localhost:{LOCAL_PORT}"


def is_ollama_alive():
    try:
        requests.get(f"{BASE_URL}/api/tags", timeout=1)
        return True
    except:
        return False

def ensure_tunnel():
    if not SSH_HOST:
        raise Exception("No SSH_HOST")
    if is_ollama_alive():
        return None
    return subprocess.Popen([
        "ssh", "-N",
        "-L", f"{LOCAL_PORT}:localhost:{REMOTE_PORT}",
        SSH_HOST
    ])

def wait_until_ready():
    for _ in range(10):
        if is_ollama_alive():
            return
        time.sleep(1)
    raise RuntimeError("Ollama not reachable")

def ensure_model(model:str):
    models = [m["name"] for m in requests.get(f"{BASE_URL}/api/tags").json().get("models", [])]
    if model not in models:
        requests.post(f"{BASE_URL}/api/pull", json={"name": model})