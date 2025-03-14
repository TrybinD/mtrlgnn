from src.problems.base import BaseSolution, BaseInstance, BaseGenerator
from src.problems import TSP
from src import problems
from inspect import getmembers, isclass
import pytest
import torch


solution_classes = [
    p[1] for p in getmembers(problems, isclass) if issubclass(p[1], BaseSolution)
]
generator_classes = [
    p[1] for p in getmembers(problems, isclass) if issubclass(p[1], BaseGenerator)
]
instance_classes = [
    p[1] for p in getmembers(problems, isclass) if issubclass(p[1], BaseInstance)
]


class TestProblems:
    def setup_class(self):
        self.params = dict()

        # TSP
        self.params[TSP] = dict()

        num_cities = 10
        locs = torch.rand((num_cities, 2))
        dist = torch.cdist(locs, locs)
        self.params[TSP][BaseInstance] = {
            "name": "test_tsp",
            "num_cities": num_cities,
            "locs": locs,
            "dist": dist,
        }

        tour = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        self.params[TSP][BaseSolution] = {"tour": tour}

    @pytest.mark.parametrize("instance_class", instance_classes)
    def test_instance(self, instance_class):
        instance_class(**self.params[instance_class.__bases__[1]][BaseInstance])

    @pytest.mark.parametrize("solution_class", solution_classes)
    def test_solution(self, solution_class):
        solution_class(**self.params[solution_class.__bases__[1]][BaseSolution])

    @pytest.mark.parametrize("generator_class", generator_classes)
    def test_generator(self, generator_class):
        generator = generator_class()
        instance = generator.sample()
        assert isinstance(instance, BaseInstance)

    @pytest.mark.parametrize("solution_class", solution_classes)
    @pytest.mark.parametrize("instance_class", instance_classes)
    def test_is_feasibel(self, solution_class, instance_class):
        if solution_class.__bases__[1] == instance_class.__bases__[1]:
            instance = instance_class(
                **self.params[instance_class.__bases__[1]][BaseInstance]
            )
            solution = solution_class(
                **self.params[solution_class.__bases__[1]][BaseSolution]
            )
            assert instance.is_feasible(solution)

    @pytest.mark.parametrize("solution_class", solution_classes)
    @pytest.mark.parametrize("instance_class", instance_classes)
    def test_total_cost(self, solution_class, instance_class):
        if solution_class.__bases__[1] == instance_class.__bases__[1]:
            instance = instance_class(
                **self.params[instance_class.__bases__[1]][BaseInstance]
            )
            solution = solution_class(
                **self.params[solution_class.__bases__[1]][BaseSolution]
            )
            assert type(instance.total_cost(solution)) is float
