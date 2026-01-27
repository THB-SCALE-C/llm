


from .google import GoogleEmbeddingModel
from .openai import OpenaiEmbeddingModel
from .openrouter import OpenrouterEmbeddingModel


def get_embedding_model(provider:str, model:str):
    if provider == "openrouter":
        _model = OpenrouterEmbeddingModel(model)
    elif provider == "openai":
        _model = OpenaiEmbeddingModel(model)
    elif provider == "google":
        _model = GoogleEmbeddingModel(model)
    else:
        raise ValueError("Embedding provider not found.")
    return _model


__all__ = ["get_embedding_model"]