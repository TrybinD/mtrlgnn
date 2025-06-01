from pathlib import Path
import random
from copy import deepcopy
from typing import Dict, List, Tuple
from src.problems.base import BaseGenerator, BaseInstance, BaseProblem, BaseSolution
import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt
import networkx as nx

from rl4co.envs import FJSPEnv


class FJSP(BaseProblem):
    pass


class FJSPSolution(BaseSolution, FJSP):
    def __init__(self,
                 actions_sequence: List[int]):
        super().__init__()
        self.actions_sequence = actions_sequence


class FJSPInstance(BaseInstance, FJSP):
    def __init__(self, 
                 name: str, 
                 start_op_per_job, 
                 end_op_per_job,
                 proc_times, 
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
                                   "pad_mask": pad_mask}, 
                                   batch_size=[1])
        
        self.n_jobs = start_op_per_job.shape[1]
        self.n_machines = proc_times.shape[1]

    def is_feasible(self, solution: FJSPSolution) -> bool:
        
        try:
            self.total_cost(solution)
            return True
        except AssertionError:
            return False

    def total_cost(self, solution: FJSPSolution) -> float:
        
        env =  FJSPEnv()
        td = env.reset(self.td_init)
        actions_sequence = deepcopy(solution.actions_sequence)

        while not td["done"].all():

            action = actions_sequence.pop(0)
            # print(action)
            td["action"] = torch.Tensor([action]).to(int)
            td = env.step(td)["next"]

        return td["time"].item()
    
    def plot_fjsp_disjunctive_graph(self, 
                                    node_size=1000, 
                                    node_color="white", 
                                    edgecolors='black',
                                    font_size=6,
                                    **kwargs):

        env = FJSPEnv()
        td = env.reset(self.td_init)

        op_name_str = "O_{j}_{op_s}({o})"
        ma_name_str = "M_{m}"

        G = nx.DiGraph()
        job_size = len(td["job_ops_adj"][0])
        G.add_node("S", type="source", pos=[-1, job_size/2])  # Фиктивный начальный узел
        G.add_node("T", type="sink", pos=[max(td["ops_sequence_order"][0]).item() + 1, job_size/2])    # Фиктивный конечный узел

        operations_for_jobs = {}
        for j in range(len(td["job_ops_adj"][0])):
            for o in range(len(td["job_ops_adj"][0][j])):
                if td["job_ops_adj"][0][j][o] == 1:
                    operations_for_jobs[o] = j

        for o in operations_for_jobs:
            j = operations_for_jobs.get(o)
            if j is not None:
                node_name = op_name_str.format(j=j, op_s=td['ops_sequence_order'][0][o], o=o)
                G.add_node(node_name, type="operation", pos=[td["ops_sequence_order"][0][o].item(), j])

        for m in range(len(td["ops_ma_adj"][0])):
            machine_node = ma_name_str.format(m=m)
            G.add_node(machine_node, type="machine", pos=[m-0.5, job_size+1])

        for j in range(len(td["job_ops_adj"][0])):
            job_ops = [o for o in range(len(td["job_ops_adj"][0][j])) if td["job_ops_adj"][0][j][o] == 1]
            sorted_ops = sorted(job_ops, key=lambda o: td["ops_sequence_order"][0][o])
            op_nodes = [op_name_str.format(j=j, op_s=td['ops_sequence_order'][0][o], o=o) for o in sorted_ops]
            
            # Добавление ребер между операциями
            for i in range(len(op_nodes) - 1):
                G.add_edge(op_nodes[i], op_nodes[i+1])
            
            # Соединение с фиктивными узлами
            if op_nodes:
                G.add_edge("S", op_nodes[0])
                G.add_edge(op_nodes[-1], "T")

        nx.draw(G, G.nodes.data("pos"), 
        with_labels=True, 
        node_size=node_size, 
        node_color=node_color, 
        font_size=font_size, 
        edgecolors=edgecolors,
        **kwargs)

        for m in range(len(td["ops_ma_adj"][0])):
            machine_edge_list = []
            for o in range(len(td["ops_ma_adj"][0][m])):
                if td["ops_ma_adj"][0][m][o] == 1:
                    j = operations_for_jobs.get(o)
                    if j is not None:
                        op_node = op_name_str.format(j=j, op_s=td['ops_sequence_order'][0][o], o=o)
                        machine_node = ma_name_str.format(m=m)
                        G.add_edge(machine_node, op_node)
                        machine_edge_list.append((machine_node, op_node))

            if machine_edge_list:
                nx.draw_networkx_edges(G, 
                                    G.nodes.data("pos"),
                                    edgelist=machine_edge_list, 
                                    edge_color=(random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)), 
                                    width=1, 
                                    style='dashed')

        plt.show()


class FJSPRL4COGenerator(BaseGenerator, FJSP):
    def __init__(self, 
                 num_jobs, 
                 num_machines,
                 min_ops_per_job,
                 max_ops_per_job,
                 min_processing_time,
                 max_processing_time,
                 min_eligible_ma_per_op,
                 max_eligible_ma_per_op, 
                 **kwargs):
        
        self.generator_params = {
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "min_ops_per_job": min_ops_per_job,
            "max_ops_per_job": max_ops_per_job,
            "min_processing_time": min_processing_time,
            "max_processing_time": max_processing_time,
            "min_eligible_ma_per_op": min_eligible_ma_per_op,
            "max_eligible_ma_per_op": max_eligible_ma_per_op
            }

    def sample(self, random_seed=None) -> FJSPInstance:

        env = FJSPEnv(generator_params=self.generator_params, seed=random_seed)

        td = env.reset(batch_size=[1])

        return FJSPInstance(name="fjsp",
                            start_op_per_job=td["start_op_per_job"], 
                            end_op_per_job=td["end_op_per_job"], 
                            proc_times=td["proc_times"], 
                            pad_mask=td["pad_mask"])


class FJSPFileGenerator(BaseGenerator, FJSP):
    def __init__(self, file: Path):

        with open(file) as f:
            lines = f.readlines()

        n_jobs, n_machines, _ = lines[0].split() # first_line - num_jobs num_machines stuff

        n_jobs = int(n_jobs)
        n_machines = int(n_machines)

        start_op_per_job, end_op_per_job, proc_times = self._decode_lines(lines[1:], n_jobs, n_machines)

        self.fjsp_instance = FJSPInstance(name="fjsp",
                                          start_op_per_job=start_op_per_job, 
                                          end_op_per_job=end_op_per_job, 
                                          proc_times=proc_times)


    def sample(self, random_seed=None) -> FJSPInstance:
        return self.fjsp_instance

    def _decode_lines(self, lines, n_jobs, n_machines):
        
        start_op_per_job = torch.zeros(1, n_jobs, dtype=int)
        end_op_per_job = torch.zeros(1, n_jobs, dtype=int)
        operations_info = [] # [(operation, machine, time)]
        
        operation_id = 0

        for job, job_operations in enumerate(lines):

            job_operations_info = job_operations.split()

            n_operations = int(job_operations_info.pop(0))

            end_op_per_job[0][job] = start_op_per_job[0][job] + n_operations - 1
            
            if job + 1 < n_jobs: # if there is next job
                start_op_per_job[0][job + 1] = end_op_per_job[0][job] + 1 # next job start op = end op of current job + 1

            while job_operations_info:
                n_machines_for_op = int(job_operations_info.pop(0))

                for _ in range(n_machines_for_op):
                    operation_machine = int(job_operations_info.pop(0)) - 1
                    operation_machine_time = float(job_operations_info.pop(0))

                    operations_info.append((operation_id, operation_machine, operation_machine_time))

                operation_id += 1

        proc_times = torch.zeros(1, n_machines, operation_id)

        for operation, machine, time in operations_info:
            proc_times[0][machine][operation] = time


        return start_op_per_job, end_op_per_job, proc_times
