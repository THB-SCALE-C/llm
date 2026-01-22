from typing import Any, Literal
from dspy.utils.callback import BaseCallback
from openai import OpenAI
from llm.lib.utils import get_provider_key
from llm.provider.base.dspy_lm_base import DspyLM
from llm.provider.base.embedding_model import BaseEmbeddingModel
from google import genai


class GoogleEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model: str = "gemini-embedding-001", key=None) -> None:
        super().__init__("google", model, key)
        self.client = genai.Client()

    def _embed(self, input: str | list[str]) -> list[list[float]]:

        result = self.client.models.embed_content(
            model=self.model,
            contents=input
        )

        if not result.embeddings:
            raise ValueError("no embeddings")

        return [e.values for e in result.embeddings if e.values]


class GoogleLM(DspyLM):
    def __init__(self, model: str = "gemma-3-27b-it", key: str | None = None, model_type: Literal['chat'] | Literal['text'] | Literal['responses'] = "chat", temperature: float | None = 0.5, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs):
        super().__init__("google", model, key, "gemini/", model_type, temperature, max_tokens, cache, callbacks,
                         num_retries, finetuning_model, launch_kwargs, train_kwargs, use_developer_role, **kwargs)


