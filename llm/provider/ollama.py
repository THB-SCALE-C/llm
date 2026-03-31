import json
from typing import Any, Literal
import dspy
from dspy.utils.callback import BaseCallback
import requests
from llm.lib.remote_ollama import ensure_model
from llm.lib.url import OLLAMA_REMOTE
from llm.lib.utils import get_logger
from llm.provider.base.dspy_lm_base import DspyLM
from llm.provider.base.embedding_model import BaseEmbeddingModel
from ollama import Client



class OllamaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str="mxbai-embed-large", key=None) -> None:
        self.client =  Client(OLLAMA_REMOTE)
        #ensure model

        print("ensuring model...")
        ensure_model(model)
        
        super().__init__(None, model, key)


    def _embed(self, input: str | list[str]) -> list[list[float]]:
        response = self.client.embed(
            model=self.model,
            input=input,
        )
        get_logger().debug(response.model_dump())
        return response.embeddings # type:ignore


class OllamaLM(DspyLM):
    def __init__(self, model: str = "llama3", key: str | None = None, model_type: Literal['chat'] | Literal['text'] | Literal['responses'] = "chat", temperature: float | None = 0.5, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: dspy.Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, create_tunnel=False, **kwargs):

        
        print("ensuring model...")
        ensure_model(model)
        api_base = OLLAMA_REMOTE
        super().__init__(None, model, key, "ollama_chat/", model_type, temperature, max_tokens, cache, callbacks,
                         num_retries, finetuning_model, launch_kwargs, train_kwargs, use_developer_role, api_base=api_base, **kwargs)
