
def get_embedding_model(provider:str, model:str):
    if provider == "openrouter":
        from .openrouter import OpenrouterEmbeddingModel
        _model = OpenrouterEmbeddingModel(model)
    elif provider == "google":
        from .google import GoogleEmbeddingModel
        _model = GoogleEmbeddingModel(model)
    elif provider == "ollama":
        from llm.provider.ollama import OllamaEmbeddingModel
        _model = OllamaEmbeddingModel(model)
    else:
        raise ValueError("Embedding provider not found.")
    return _model


__all__ = ["get_embedding_model"]