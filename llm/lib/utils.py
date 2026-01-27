from logging import Logger as _Logger
from types import TracebackType
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = Literal['openai', 'openrouter', 'ollama', 'google']

DEBUG_MSG_TEMPLATE = """\n===DEBUG===\n{msg}\n"""

class Logger(_Logger):
    def __init__(self, name: str, level: int | str = 4) -> None:
        super().__init__(name, level)

    def debug(self, msg: object, *args: object, exc_info: None | bool | tuple[type[BaseException], BaseException, TracebackType | None] | tuple[None, None, None] | BaseException = None, stack_info: bool = False, stacklevel: int = 1, extra: os.Mapping[str, object] | None = None) -> None: # type: ignore
        msg = DEBUG_MSG_TEMPLATE.format(msg=msg)
        print(msg)
        return super().debug(msg, *args, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)


def get_provider_key(provider: PROVIDER):
    ENV_KEY = f"{provider.capitalize()}_API_KEY"
    api_key = os.getenv(ENV_KEY, None)
    if not api_key:
        raise ValueError(f"Please provide {ENV_KEY} in .env")
    return api_key


def get_logger(name:str="logger"):
    return Logger(name)