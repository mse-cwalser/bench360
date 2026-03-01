from abc import ABC, abstractmethod
from typing import Literal

class BaseTask(ABC):
    """
    Abstract base class for all NLP tasks.
    """
    api_type: Literal["completion", "chat_completion"] = "completion"

    @abstractmethod
    def generate_prompts(self, num_examples: int) -> tuple[list[str], list[str]]:
        """
        Should return a list of prompts and reference answers.
        """
        pass

    @abstractmethod
    def quality_metrics(self, generated: str, reference: str) -> dict[str, float]:
        """
        Should return a dictionary of task-specific evaluation metrics.
        """
        pass
