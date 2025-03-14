from src.problems.base import BaseSolution, BaseInstance, BaseGenerator
from torch import Tensor


class BaseVRPSolution(BaseSolution):
    pass


class BaseVRPInstance(BaseInstance):
    def __init__(self, name: str, num_cities: int, locs: Tensor, dist: Tensor):
        super().__init__(name)
        assert dist.shape[0] == dist.shape[1]
        assert dist.shape[0] == num_cities
        self.num_cities = num_cities
        self.locs = locs
        self.dist = dist


class BaseVRPGenerator(BaseGenerator):
    pass
