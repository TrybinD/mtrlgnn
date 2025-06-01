from rl4co.envs import FJSPEnv
from rl4co.envs.scheduling.fjsp.generator import FJSPGenerator

import torch

class FJSPGeneratorWithGPM(FJSPGenerator):
    """FJSP Generator with General Purpose Machine"""

    def __init__(self, 
                 num_jobs = 10, 
                 num_machines = 5, 
                 min_ops_per_job = 4, 
                 max_ops_per_job = 6, 
                 min_processing_time = 1, 
                 max_processing_time = 20, 
                 min_eligible_ma_per_op = 1, 
                 max_eligible_ma_per_op = None, 
                 same_mean_per_op = True, 
                 **unused_kwargs):
        
        super().__init__(num_jobs, 
                         num_machines, 
                         min_ops_per_job, 
                         max_ops_per_job, 
                         min_processing_time, 
                         max_processing_time, 
                         min_eligible_ma_per_op, 
                         max_eligible_ma_per_op, 
                         same_mean_per_op, 
                         **unused_kwargs)

    def _generate(self, batch_size):
        td = super()._generate(batch_size)

        bs, n_machines, n_ops = td["proc_times"].shape

        general_purpose_machine = torch.ones(size=(bs, 1, n_ops), 
                                             dtype=td["proc_times"].dtype, 
                                             device=td["proc_times"].device) * self.max_processing_time

        td["proc_times"] = torch.cat([td["proc_times"], general_purpose_machine], dim=1)

        return td

        
class FJSPEnvMOPM(FJSPEnv):
    """FJSP with Maximum Operation Per Machine"""
    def __init__(self, generator_params = ..., check_mask = False, stepwise_reward = False, **kwargs):

        generator_params = {**generator_params}
        self.max_ops_processed = generator_params.pop("max_ops_processed")

        generator = FJSPGeneratorWithGPM(**generator_params)

        super().__init__(generator=generator, mask_no_ops=True, check_mask=check_mask, stepwise_reward=stepwise_reward, **kwargs)

    def _reset(self, td = None, batch_size=None):
        td = super()._reset(td, batch_size)

        ma_ops_processed_left = torch.ones_like(td["busy_until"]) * self.max_ops_processed

        td["ma_ops_processed_left"] = ma_ops_processed_left

        td["ma_ops_processed_left"][:, -1] = 1e6

        return td.to(self.device)
    
    def _step(self, td):

        td = td.to(self.device)

        # test if we can use new action

        n_batches, n_jobs = td["end_op_per_job"].shape

        ma_ops_processed_left = td["ma_ops_processed_left"]

        n_machines = ma_ops_processed_left.size(1)

        machines = (td["action"] - 1) % n_machines

        ma_ops_processed_left[torch.arange(ma_ops_processed_left.size(0), device=self.device)[td["action"] > 0], machines[td["action"] > 0]] -= 1

        assert (ma_ops_processed_left >= 0).all()

        td = super()._step(td)

        td["ma_ops_processed_left"] = ma_ops_processed_left

        availible_machines_mask = (ma_ops_processed_left > 0)

        new_mask = torch.concat([torch.tensor([True], device=self.device).bool().repeat(n_batches, 1), availible_machines_mask.repeat(1, n_jobs)], dim=1)

        new_mask = td["action_mask"] * new_mask

        all_false_rows = ~new_mask.any(dim=1)

        new_mask[all_false_rows, 0] = True

        td["action_mask"] = new_mask

        return td


class FJSPEnvMTPM(FJSPEnv):
    """FJSP with Maximum Time Per Machine"""
    def __init__(self, generator_params = ..., check_mask = False, stepwise_reward = False, **kwargs):

        generator_params = {**generator_params}
        self.max_time_worked = generator_params.pop("max_time_worked")

        generator = FJSPGeneratorWithGPM(**generator_params)

        super().__init__(generator=generator, mask_no_ops=True, check_mask=check_mask, stepwise_reward=stepwise_reward, **kwargs)

    def _reset(self, td = None, batch_size=None):
        td = super()._reset(td, batch_size)

        ma_time_left = torch.ones_like(td["busy_until"]) * self.max_time_worked

        td["ma_time_left"] = ma_time_left

        td["ma_time_left"][:, -1] = 1e6

        return td.to(self.device)
    
    def _step(self, td):

        td = td.to(self.device)

        # test if we can use new action

        n_batches, n_jobs = td["end_op_per_job"].shape

        ma_time_left = td["ma_time_left"]

        n_machines = ma_time_left.size(1)

        machines = (td["action"] - 1) % n_machines

        jobs = (td["action"] - 1) // n_machines

        ops = td["next_op"][torch.arange(n_batches), jobs]

        ops_time = td["proc_times"][torch.arange(n_batches), machines, ops]

        ma_time_left[torch.arange(ma_time_left.size(0), device=self.device)[td["action"] > 0], machines[td["action"] > 0]] -= ops_time.flatten()[td["action"] > 0]

        assert (ma_time_left >= 0).all()

        td = super()._step(td)

        td["ma_time_left"] = ma_time_left

        batches, available_jobs, available_machines = torch.where(td["action_mask"][:, 1:].reshape(n_batches, n_jobs, n_machines))

        ops = td["next_op"][batches, available_jobs]

        ops_time = td["proc_times"][batches, available_machines, ops]

        available_ops_mask = (ma_time_left[batches, available_machines] > ops_time)

        new_mask = torch.ones(size=(n_batches, n_jobs, n_machines), dtype=bool, device=self.device)

        new_mask[batches, available_jobs, available_machines] = available_ops_mask

        new_mask = new_mask.reshape(n_batches, -1)

        new_mask = torch.concat([torch.tensor([True], device=self.device).bool().repeat(n_batches, 1), new_mask], dim=1)

        new_mask *= td["action_mask"]

        all_false_rows = ~new_mask.any(dim=1)

        new_mask[all_false_rows, 0] = True
        
        td["action_mask"] = new_mask

        return td
