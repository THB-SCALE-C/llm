from abc import ABC, abstractmethod
from pydantic import BaseModel
from llm.lib.utils import PROVIDER, get_provider_key


class Embedding(BaseModel):
    vector: list[float]
    text: str

    def __repr__(self) -> str:
        return f"{self.text[:50]}:\t{self.vector}"

    def __str__(self) -> str:
        return f"{self.text[:50]}:\t{self.vector}"


class BaseEmbeddingModel(ABC):
    def __init__(self, provider: PROVIDER, model: str, key=None) -> None:
        self.model = model
        self.provider = provider
        self.key = key if key else get_provider_key(provider)

    @abstractmethod
    def _embed(self, input: str | list[str]) -> list[list[float]]:
        pass

    def embed(self, input: str | list[str], step: int | None = None) -> list[Embedding]:
        _all = []
        if step and isinstance(input, list):
            for i in range(0, len(input), step):
                _texts = input[i:i+step]
                try:
                    embeddings = self._embed(_texts)
                except Exception as e:
                    print(e)
                    embeddings = []
                _all.extend(embeddings)
        elif isinstance(input, str):
            input = [input]
            try:
                _all = self._embed(input)
            except Exception as e:
                print(e)
                _all = []
        return [Embedding(vector=e, text=t)
                for e, t in zip(_all, input)]
