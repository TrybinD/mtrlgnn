from abc import ABC, abstractmethod
from src.problems.base import BaseInstance, BaseSolution


class BaseDataset(ABC):
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[BaseInstance, BaseSolution]:
        pass

    @abstractmethod
    def __len__(self):
        pass
