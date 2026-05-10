# Bug fix: expose all three active pipelines
from .chat_pipeline import ChatPipeline
from .deep_research_pipeline import DeepResearchPipeline
from .study_pipeline import StudyPipeline

__all__ = [
    "ChatPipeline",
    "DeepResearchPipeline",
    "StudyPipeline",
]
