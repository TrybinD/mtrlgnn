
from abc import abstractmethod
from typing import Tuple
from rl4co.envs import FJSPEnv
import torch


from src.problems.jsp.fjsp import FJSP, FJSPInstance, FJSPSolution
from src.solvers.base import BaseSolver


class FJSPBaseSolver(BaseSolver, FJSP):

    def solve(self, instance: FJSPInstance) -> FJSPSolution:
        
        env =  FJSPEnv()
        td = env.reset(instance.td_init)
        solution = []

        while not td["done"].all():
            action, job, machine = self.get_action(td, )
            td["action"] = torch.Tensor([action]).to(int)

            solution.append(action)

            td = env.step(td)["next"]

        return FJSPSolution(actions_sequence=solution)
    
    @abstractmethod
    def get_action(self, td) -> Tuple[int, int, int]:
        pass
    