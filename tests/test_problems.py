from src.problems.base import BaseSolution, BaseInstance, BaseGenerator
from src.solvers.base import BaseSolver, BaseRLSolver
from src import solvers
from src import problems
from inspect import getmembers, isclass
import pytest


solver_classes = [
    s[1] for s in getmembers(solvers, isclass) if issubclass(s[1], BaseSolver)
]
generators_classes = [
    p[1] for p in getmembers(problems, isclass) if issubclass(p[1], BaseGenerator)
]


test_set = dict()
for g in generators_classes:
    for s in solver_classes:
        if g.__bases__[1] == s.__bases__[1]:
            test_set[g.__name__ + s.__name__] = (g, s)


@pytest.mark.parametrize("test_key", test_set)
def test_solver(test_key):
    generator_class, solver_class = test_set[test_key]
    generator = generator_class()
    instance = generator.sample()
    solver = solver_class()
    if issubclass(solver_class, BaseRLSolver):
        solver = solver_class(
            train_data_size=1,
            val_data_size=1,
            test_data_size=1,
            batch_size=1,
            max_epochs=1,
        )
        solver.fit()
    solution = solver.solve(instance)
    assert issubclass(type(solution), BaseSolution)


@pytest.mark.parametrize("test_key", test_set)
def test_instance(test_key):
    generator_class, solver_class = test_set[test_key]
    generator = generator_class()
    instance = generator.sample()
    assert issubclass(type(instance), BaseInstance)
    solver = solver_class()
    solution = solver.solve(instance)
    instance.is_feasible(solution)
    instance.total_cost(solution)
