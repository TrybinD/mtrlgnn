import collections
from typing import Tuple
from ortools.sat.python import cp_model
from rl4co.envs import FJSPEnv
import torch

from src.solvers.jsp.fjsp.base import FJSPBaseSolver
from src.problems.jsp.fjsp import FJSPInstance, FJSPSolution



class FJSPORToolsSolver(FJSPBaseSolver):

    def __init__(self, time_limit: int):
        self.time_limit = time_limit

    
    def solve(self, instance: FJSPInstance) -> FJSPSolution:


        or_tools_solution = self._solve_ortools(instance)
        
        env =  FJSPEnv()
        td = env.reset(instance.td_init)
        solution = []

        while not td["done"].all():
            action, job, machine, or_tools_solution = self.get_action(td, or_tools_solution)
            td["action"] = torch.Tensor([action]).to(int)

            solution.append(action)

            td = env.step(td)["next"]

        return FJSPSolution(actions_sequence=solution)
    
    def get_action(self, td, or_tools_solution) -> Tuple[int, int, int]:
        
        n_machines = td["proc_times"].shape[1]
        time = td["time"].item()

        start_time, machine, job = or_tools_solution[0]

        action = 1 + job*n_machines + machine

        if time == start_time:
            or_tools_solution = or_tools_solution[1:]
            return action, job, machine, or_tools_solution
        
        elif time < start_time:
            free_machines = torch.where(td["busy_until"][0] <= time)[0]

            next_job_free_machine = []

            for free_machine in free_machines:
                blocking_jobs = set()
                for i, (_, pot_machine, pot_job) in enumerate(sorted(or_tools_solution)):
                    if pot_machine == free_machine: 
                        if pot_job not in blocking_jobs:
                            next_job_free_machine.append((pot_machine, pot_job, i))
                        break
                    blocking_jobs.add(pot_job)

            for next_machine, next_job, i in next_job_free_machine:
                pot_action = 1 + next_job*n_machines + next_machine

                if td["action_mask"][0, pot_action]:
                    return pot_action, next_job, next_machine, or_tools_solution[:i] + or_tools_solution[i+1:]


            return 0, -1, -1, or_tools_solution[:]
        
        raise Exception("aaaaa")
            
    def _drop_used_actions(self, or_tools_solution, job, machine):

        for i, (_, machine_i, job_i) in enumerate(or_tools_solution):

            if machine==machine_i and job==job_i:
                return or_tools_solution[:i] + or_tools_solution[i+1:]

        return or_tools_solution[:]


    def _solve_ortools(self, instance: FJSPInstance) -> FJSPSolution:

        jobs = self._get_jobs_info(instance.td_init)

        num_jobs = len(jobs)
        all_jobs = range(num_jobs)

        num_machines = instance.n_machines
        all_machines = range(num_machines)

        model = cp_model.CpModel()

        horizon = 0
        for job in jobs:
            for task in job:
                max_task_duration = 0
                for alternative in task:
                    max_task_duration = max(max_task_duration, alternative[0])
                horizon += max_task_duration


        intervals_per_resources = collections.defaultdict(list)
        starts = {}  # indexed by (job_id, task_id).
        presences = {}  # indexed by (job_id, task_id, alt_id).
        job_ends: list[cp_model.IntVar] = []


        for job_id in all_jobs:
            job = jobs[job_id]
            num_tasks = len(job)
            previous_end = None
            for task_id in range(num_tasks):
                task = job[task_id]

                min_duration = task[0][0]
                max_duration = task[0][0]

                num_alternatives = len(task)
                all_alternatives = range(num_alternatives)

                for alt_id in range(1, num_alternatives):
                    alt_duration = task[alt_id][0]
                    min_duration = min(min_duration, alt_duration)
                    max_duration = max(max_duration, alt_duration)

                # Create main interval for the task.
                suffix_name = f"_j{job_id}_t{task_id}"
                start = model.new_int_var(0, horizon, "start" + suffix_name)
                duration = model.new_int_var(
                    min_duration, max_duration, "duration" + suffix_name
                )
                end = model.new_int_var(0, horizon, "end" + suffix_name)
                interval = model.new_interval_var(
                    start, duration, end, "interval" + suffix_name
                )

                # Store the start for the solution.
                starts[(job_id, task_id)] = start

                # Add precedence with previous task in the same job.
                if previous_end is not None:
                    model.add(start >= previous_end)
                previous_end = end

                # Create alternative intervals.
                if num_alternatives > 1:
                    l_presences = []
                    for alt_id in all_alternatives:
                        alt_suffix = f"_j{job_id}_t{task_id}_a{alt_id}"
                        l_presence = model.new_bool_var("presence" + alt_suffix)
                        l_start = model.new_int_var(0, horizon, "start" + alt_suffix)
                        l_duration = task[alt_id][0]
                        l_end = model.new_int_var(0, horizon, "end" + alt_suffix)
                        l_interval = model.new_optional_interval_var(
                            l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                        )
                        l_presences.append(l_presence)

                        # Link the primary/global variables with the local ones.
                        model.add(start == l_start).only_enforce_if(l_presence)
                        model.add(duration == l_duration).only_enforce_if(l_presence)
                        model.add(end == l_end).only_enforce_if(l_presence)

                        # Add the local interval to the right machine.
                        intervals_per_resources[task[alt_id][1]].append(l_interval)

                        # Store the presences for the solution.
                        presences[(job_id, task_id, alt_id)] = l_presence

                    # Select exactly one presence variable.
                    model.add_exactly_one(l_presences)
                else:
                    intervals_per_resources[task[0][1]].append(interval)
                    presences[(job_id, task_id, 0)] = model.new_constant(1)

            if previous_end is not None:
                job_ends.append(previous_end)

        for machine_id in all_machines:
            intervals = intervals_per_resources[machine_id]
            if len(intervals) > 1:
                model.add_no_overlap(intervals)

        # Makespan objective
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, job_ends)
        model.minimize(makespan)

        # Solve model
        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = self.time_limit

        status = solver.solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = []

            for job_id in all_jobs:
                for task_id, task in enumerate(jobs[job_id]):
                    start_value = solver.value(starts[(job_id, task_id)])
                    machine: int = -1
                    for alt_id, alt in enumerate(task):
                        if solver.boolean_value(presences[(job_id, task_id, alt_id)]):
                            duration, machine = alt

                            solution.append([start_value, machine, job_id])

            return sorted(solution)

        else:
            raise AssertionError("No FEASIBLE solution")


    def _get_jobs_info(self, td_init): 
        jobs = []

        for start, end in zip(td_init["start_op_per_job"][0], td_init["end_op_per_job"][0]):
            operations = []

            for operation in range(start, end+1):
                operation_alternatives = []
                
                for machine_id, operation_time in enumerate(td_init["proc_times"][0, :, operation]):
                    if operation_time > 0:
                        operation_alternatives.append((int(operation_time.item()), machine_id))

                operations.append(operation_alternatives)

            jobs.append(operations)

        return jobs