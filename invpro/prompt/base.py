from abc import ABC, abstractmethod

class PromptFormatter(ABC):

    @abstractmethod
    def format(self, **kwargs) -> str:
        pass

    @abstractmethod
    def format_answer(self, answer: str) -> str:
        pass

    @abstractmethod
    def extract_answer(self, response: str, **kwargs) -> str:
        pass