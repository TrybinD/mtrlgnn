from src.problems import TSP, TSPInstance
from src.problems.base import BaseSolution
from src import solvers
from inspect import getmembers, isclass
import torch
import pytest

solver_classes = [s[1] for s in getmembers(solvers, isclass)]


class TestSolvers:
    def setup_class(self):
        num_cities = 10
        locs = torch.rand((num_cities, 2))
        dist = torch.cdist(locs, locs)
        self.instance = dict()
        self.instance[TSP] = TSPInstance(
            name="test_tsp", num_cities=num_cities, locs=locs, dist=dist
        )

    @pytest.mark.parametrize("solver_class", solver_classes)
    def test_solve(self, solver_class):
        solver = solver_class()
        solution = solver.solve(self.instance[solver_class.__bases__[1]])
        assert isinstance(solution, BaseSolution)
