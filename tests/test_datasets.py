from src.problems.base import BaseSolution, BaseInstance
from inspect import getmembers, isclass
import pytest
from src import datasets

dataset_classes = [d[1] for d in getmembers(datasets, isclass)]


@pytest.mark.parametrize("dataset_class", dataset_classes)
def test_dataset(dataset_class):
    dataset = dataset_class()
    assert len(dataset) > 0
    for idx in range(len(dataset)):
        instance, solution = dataset[idx]
        assert isinstance(instance, BaseInstance)
        if solution is not None:
            assert isinstance(solution, BaseSolution)
            assert instance.is_feasible(solution)
