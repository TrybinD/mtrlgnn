from src.problems.vrp.base import BaseVRPInstance, BaseVRPSolution, BaseVRPGenerator
from src.problems.base import BaseProblem
import torch
from torch import Tensor


class TSP(BaseProblem):
    pass


class TSPSolution(BaseVRPSolution, TSP):
    def __init__(self, tour: Tensor):
        super().__init__()
        self.tour = tour


class TSPInstance(BaseVRPInstance, TSP):
    def is_feasible(self, solution):
        cond = (
            len(solution.tour) - 1 == self.num_cities,
            len(torch.unique(solution.tour)) == self.num_cities,
            set(solution.tour.tolist()) == set(range(self.num_cities)),
        )
        return all(cond)

    def total_cost(self, solution):
        cost = self.dist[solution.tour[1:], solution.tour[:-1]]
        cost = cost.sum().item()
        return cost


class TSP2DUniformGenerator(BaseVRPGenerator, TSP):
    def __init__(self, num_cities: int = 20, min_: int = 0, max_: int = 1, p: int = 2):
        assert min_ < max_
        self.num_cities = num_cities
        self.min_ = min_
        self.max_ = max_
        self.p = p

    def sample(self, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
        locs = (torch.rand((self.num_cities, 2)) + self.min_) * self.max_
        dist = torch.cdist(locs, locs, p=self.p)
        return TSPInstance('2d_uniform', self.num_cities, locs=locs, dist=dist)
