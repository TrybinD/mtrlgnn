
from typing import Tuple

import numpy as np
import torch
from src.solvers.jsp.fjsp.base import FJSPBaseSolver



class FJSPFIFOSolver(FJSPBaseSolver):
    """Selects the first free job and assigns it to the available machine with the shortest processing time"""
    
    def get_action(self, td) -> Tuple[int, int, int]:

        n_machines = td["proc_times"].shape[1]

        action_idx = np.argmax(td["action_mask"][0]).item()

        job_idx = (action_idx - 1) // n_machines

        next_operation_index = td["next_op"][0][job_idx].item()

        operation_proc_times = td["proc_times"][0][:, next_operation_index]

        job_machine_mask = td["action_mask"][0][(1+job_idx*n_machines): (1+(job_idx+1)*n_machines)]

        machine_idx = np.argmin(operation_proc_times + torch.inf*(1-job_machine_mask.to(int))).item()

        action = 1 + job_idx*n_machines + machine_idx

        return action, job_idx, machine_idx



class FJSPMOPNRSolver(FJSPBaseSolver):
    """Selecting the candidate operation with the most remaining successors and a machine with the shortest processing time which can immediately process it"""
    
    def get_action(self, td) -> Tuple[int, int, int]:

        n_machines = td["proc_times"].shape[1]

        available_jobs, available_machines = torch.where(td["action_mask"][0, 1:].reshape(-1, n_machines))

        available_ops = td["next_op"][0, available_jobs]

        last_ops = td["end_op_per_job"][0, available_jobs]

        n_ops_remain = last_ops - available_ops

        job_idx = available_jobs[np.argmax(n_ops_remain)]

        next_operation_index = td["next_op"][0][job_idx].item()

        operation_proc_times = td["proc_times"][0][:, next_operation_index]

        job_machine_mask = td["action_mask"][0][(1+job_idx*n_machines): (1+(job_idx+1)*n_machines)]

        machine_idx = np.argmin(operation_proc_times + torch.inf*(1-job_machine_mask.to(int))).item()

        action = 1 + job_idx*n_machines + machine_idx

        return action, job_idx, machine_idx
    

class FJSPSTPSolver(FJSPBaseSolver):
    """Selecting the compatible operation-machine pair with the shortest processing time"""
    
    def get_action(self, td) -> Tuple[int, int, int]:

        n_machines = td["proc_times"].shape[1]

        available_jobs, available_machines = torch.where(td["action_mask"][0, 1:].reshape(-1, n_machines))

        available_ops = td["next_op"][0, available_jobs]

        times = td["proc_times"][0, available_machines, available_ops]

        idx = np.argmin(times)

        job_idx = available_jobs[idx]

        machine_idx = available_machines[idx]

        action = 1 + job_idx*n_machines + machine_idx

        return action, job_idx, machine_idx
    

class FJSPMWKRSolver(FJSPBaseSolver):
    """Selecting the candidate operation with the most remaining successor average processing time and a machine with the shortest processing time which can immediately process it."""
    
    def get_action(self, td) -> Tuple[int, int, int]:

        operation_mean_time = td["proc_times"][0].clone()
        operation_mean_time[operation_mean_time == 0] = torch.nan
        operation_mean_time = operation_mean_time.nanmean(dim=0)

        n_machines = td["proc_times"].shape[1]

        available_jobs, available_machines = torch.where(td["action_mask"][0, 1:].reshape(-1, n_machines))

        available_ops = td["next_op"][0, available_jobs]

        last_ops = td["end_op_per_job"][0, available_jobs]

        ops_remain = list(map(torch.arange, available_ops, last_ops+1))

        work_remain = torch.Tensor(list(map(lambda x: operation_mean_time[x].sum(),  ops_remain)))

        job_idx = available_jobs[np.argmax(work_remain)]

        next_operation_index = td["next_op"][0][job_idx].item()

        operation_proc_times = td["proc_times"][0][:, next_operation_index]

        job_machine_mask = td["action_mask"][0][(1+job_idx*n_machines): (1+(job_idx+1)*n_machines)]

        machine_idx = np.argmin(operation_proc_times + torch.inf*(1-job_machine_mask.to(int))).item()

        action = 1 + job_idx*n_machines + machine_idx

        return action, job_idx, machine_idx
        