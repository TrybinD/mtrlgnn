from __future__ import annotations
from abc import ABC, abstractmethod


class BaseSolution(ABC):
    pass


class BaseInstance(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def is_feasible(self, solution: BaseSolution) -> bool:
        pass

    @abstractmethod
    def total_cost(self, solution: BaseSolution) -> float:
        pass


class BaseGenerator(ABC):
    @abstractmethod
    def sample(self, random_seed=None) -> BaseInstance:
        pass


class BaseProblem:
    pass
