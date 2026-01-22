from typing import Any, Literal
from dspy.utils.callback import BaseCallback
from openai import OpenAI
from llm.lib.utils import get_provider_key
from llm.provider.base.dspy_lm_base import DspyLM
from llm.provider.base.embedding_model import BaseEmbeddingModel
import dspy


class OpenaiEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str, key=None) -> None:
        super().__init__("openai", model, key)

    def _embed(self, input: str | list[str]) -> list[list[float]]:
        _client = OpenAI(api_key=self.key)
        response = _client.embeddings.create(
            input=input,
            model=self.model
        )
        return [d.embedding for d in response.data]


class OpenaiLM(DspyLM):
    def __init__(self, model: str, key: str | None = None, model_type: Literal['chat'] | Literal['text'] | Literal['responses'] = "chat", temperature: float | None = 0.5, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs):
        super().__init__("openai", model, key, "openai/", model_type, temperature, max_tokens, cache, callbacks,
                         num_retries, finetuning_model, launch_kwargs, train_kwargs, use_developer_role, **kwargs)
