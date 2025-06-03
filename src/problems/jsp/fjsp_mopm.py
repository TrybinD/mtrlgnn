from copy import deepcopy
from typing import List
from src.problems.base import BaseGenerator, BaseInstance, BaseProblem, BaseSolution
import torch
from tensordict import TensorDict

from src.problems.jsp.rl4co_extentions.fjsp_assembly_constraints import FJSPEnvMOPM


class FJSPMOPM(BaseProblem):
    """FJSP with Maximum Operation Per Machine"""
    pass


class FJSPMOPMSolution(BaseSolution, FJSPMOPM):
    def __init__(self,
                 actions_sequence: List[int]):
        super().__init__()
        self.actions_sequence = actions_sequence


class FJSPMOPMInstance(BaseInstance, FJSPMOPM):
    def __init__(self, 
                 name: str, 
                 start_op_per_job, 
                 end_op_per_job,
                 proc_times, 
                 ma_ops_processed_left,
                 pad_mask = None):
        super().__init__(name)

        if len(start_op_per_job.shape) == 1:
            start_op_per_job = start_op_per_job.unsqueeze(0)

        if len(end_op_per_job.shape) == 1:
            end_op_per_job = end_op_per_job.unsqueeze(0)

        if len(proc_times.shape) == 2:
            proc_times = proc_times.unsqueeze(0)
        
        if pad_mask is None:
            pad_mask = torch.zeros((1, proc_times.shape[2]), dtype=torch.bool)
        
        self.td_init = TensorDict({"end_op_per_job": end_op_per_job, 
                                   "start_op_per_job": start_op_per_job, 
                                   "proc_times": proc_times, 
                                   "pad_mask": pad_mask, 
                                   "ma_ops_processed_left": ma_ops_processed_left}, 
                                   batch_size=[1])
        
        self.n_jobs = start_op_per_job.shape[1]
        self.n_machines = proc_times.shape[1]

    def is_feasible(self, solution: FJSPMOPMSolution) -> bool:
        
        try:
            self.total_cost(solution)
            return True
        except AssertionError:
            return False

    def total_cost(self, solution: FJSPMOPMSolution) -> float:
        
        env =  FJSPEnvMOPM()
        td = env.reset(self.td_init)
        actions_sequence = deepcopy(solution.actions_sequence)

        while not td["done"].all():

            action = actions_sequence.pop(0)
            td["action"] = torch.Tensor([action]).to(int)
            td = env.step(td)["next"]

        return td["time"].item()


class FJSPMOPMRL4COGenerator(BaseGenerator, FJSPMOPM):
    def __init__(self, 
                 num_jobs, 
                 num_machines,
                 min_ops_per_job,
                 max_ops_per_job,
                 min_processing_time,
                 max_processing_time,
                 min_eligible_ma_per_op,
                 max_eligible_ma_per_op, 
                 max_ops_limit, 
                 min_ops_limit,
                 **kwargs):
        
        self.generator_params = {
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "min_ops_per_job": min_ops_per_job,
            "max_ops_per_job": max_ops_per_job,
            "min_processing_time": min_processing_time,
            "max_processing_time": max_processing_time,
            "min_eligible_ma_per_op": min_eligible_ma_per_op,
            "max_eligible_ma_per_op": max_eligible_ma_per_op,
            "max_ops_limit": max_ops_limit,
            "min_ops_limit": min_ops_limit
            }

    def sample(self, random_seed=None) -> FJSPMOPMInstance:

        env = FJSPEnvMOPM(generator_params=self.generator_params, seed=random_seed)

        td = env.reset(batch_size=[1])

        return FJSPMOPMInstance(name="fjsp_mopm",
                            start_op_per_job=td["start_op_per_job"], 
                            end_op_per_job=td["end_op_per_job"], 
                            proc_times=td["proc_times"], 
                            ma_ops_processed_left=td["ma_ops_processed_left"],
                            pad_mask=td["pad_mask"])
