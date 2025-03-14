from src.problems.vrp.base import BaseVRPInstance, BaseVRPSolution, BaseVRPGenerator
import torch
from torch import Tensor


class TSPProblem:
    pass


class TSPSolution(BaseVRPSolution, TSPProblem):
    def __init__(self, route: Tensor):
        super().__init__()
        self.route = route


class TSPInstance(BaseVRPInstance, TSPProblem):
    def is_feasible(self, solution):
        cond = (
            len(solution.route) - 1 == self.dist.shape[0],
            len(torch.unique(solution.route)) == self.num_cities,
        )
        return all(cond)

    def total_cost(self, solution):
        cost = self.dist[solution.route[1:], solution.route[:-1]]
        cost = cost.sum()
        return cost


class TSP2DUniformGenerator(BaseVRPGenerator, TSPProblem):
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
