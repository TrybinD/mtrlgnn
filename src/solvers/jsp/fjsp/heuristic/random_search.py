from typing import Tuple
from rl4co.envs.scheduling import FJSPEnv

import numpy as np
import torch
from src.problems.jsp.fjsp import FJSPInstance, FJSPSolution
from src.solvers.jsp.fjsp.base import FJSPBaseSolver



class FJSPRandomSolver(FJSPBaseSolver):
    """Selects random next action multiple times and then take best"""

    def __init__(self, n_search: int):
        super().__init__()
        self.n_search = n_search
    
    def solve(self, instance: FJSPInstance) -> FJSPSolution:
        
        env =  FJSPEnv()
        solutions = {}

        for _ in range(self.n_search):
            solution = []

            td = env.reset(instance.td_init)

            while not td["done"].all():
                action, job, machine = self.get_action(td)
                td["action"] = torch.Tensor([action]).to(int)

                solution.append(action)

                td = env.step(td)["next"]

            result = td["time"].item()

            solutions[result] = solution

        best_time = min(solutions.keys())
        best_solution = solutions[best_time]

        return FJSPSolution(actions_sequence=best_solution)
    
    def get_action(self, td) -> Tuple[int, int, int]:

        n_machines = td["proc_times"].shape[1]

        actions = torch.nonzero(td["action_mask"][0])

        action = actions[torch.randint(len(actions), (1,))].item()

        job_idx = (action - 1) // n_machines

        machine = (action - 1) % n_machines

        return action, job_idx, machine