class MiniNotebookLMError(Exception):
    """Base exception."""
    pass

class IngestionError(MiniNotebookLMError):
    """File processing failed."""
    pass

class RetrievalError(MiniNotebookLMError):
    """Search failed."""
    pass

class GenerationError(MiniNotebookLMError):
    """LLM call failed."""
    pass

class StorageError(MiniNotebookLMError):
    """Database operation failed."""
    pass

class ConfigurationError(MiniNotebookLMError):
    """Invalid configuration."""
    pass