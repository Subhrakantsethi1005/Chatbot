"""Production pipeline that updates, validates, versions and monitors the
chatbot's knowledge base. See docs/PIPELINE.md."""

from .config import PipelineConfig, load_config
from .control import ControlPlane
from .pipeline import KnowledgeBasePipeline, UpdateReport
from .service import ChatResponse, ChatService

__all__ = ["ChatResponse", "ChatService", "ControlPlane", "KnowledgeBasePipeline", "PipelineConfig",
           "UpdateReport", "load_config"]
