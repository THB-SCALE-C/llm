from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = Literal['openai', 'openrouter', 'ollama', 'google']


def get_provider_key(provider: PROVIDER):
    ENV_KEY = f"{provider.capitalize()}_API_KEY"
    api_key = os.getenv(ENV_KEY, None)
    if not api_key:
        raise ValueError(f"Please provide {ENV_KEY} in .env")
    return api_key