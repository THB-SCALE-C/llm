import json
from typing import Any, Literal
import dspy
from dspy.utils.callback import BaseCallback
from openai import OpenAI
import requests
from llm.lib.url import OPENROUTER_BASE_URL
from llm.lib.utils import get_logger, get_provider_key
from llm.provider.base.dspy_lm_base import DspyLM
from llm.provider.base.embedding_model import BaseEmbeddingModel


class OpenrouterEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str="sentence-transformers/all-minilm-l12-v2", key=None) -> None:
        super().__init__("openrouter", model, key)

    def _embed(self, input: str | list[str]) -> list[list[float]]:
        config = {
            "url": f"{OPENROUTER_BASE_URL}/embeddings",
            "headers": {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json"
            },

        }
        response = requests.post(
            **config,
            data=json.dumps({
                "model": self.model,
                "input": input,
                "encoding_format": "float"
            })
        )
        get_logger().debug(response.json())
        return [d["embedding"] for d in response.json()["data"]]


class OpenrouterLM(DspyLM):
    def __init__(self, model: str = "openai/gpt-oss-120b", key: str | None = None, model_type: Literal['chat'] | Literal['text'] | Literal['responses'] = "chat", temperature: float | None = 0.5, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: dspy.Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs):

        api_base = "https://openrouter.ai/api/v1"
        super().__init__("openrouter", model, key, "openrouter/", model_type, temperature, max_tokens, cache, callbacks,
                         num_retries, finetuning_model, launch_kwargs, train_kwargs, use_developer_role, api_base=api_base, **kwargs)
