from typing import Any, Literal
import dspy
from dspy.utils.callback import BaseCallback
from llm.lib.utils import PROVIDER, get_provider_key


class DspyLM(dspy.LM):
    def __init__(self,
                 provider: PROVIDER,
                 model: str,
                 key: str | None = None, model_prefix: str = "",
                 model_type: Literal['chat'] | Literal['text'] | Literal['responses'] = "chat", temperature: float | None = 0.5, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs):
        if key is None:
            key = get_provider_key(provider)
        super().__init__(f"{model_prefix}{model}", model_type, temperature, max_tokens, cache, callbacks,
                         num_retries, None, finetuning_model, launch_kwargs, train_kwargs, use_developer_role, **kwargs)
