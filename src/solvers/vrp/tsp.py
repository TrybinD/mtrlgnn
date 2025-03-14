from src.solvers.base import BaseSolver, BaseRLSolver
from src.problems.vrp.tsp import TSPSolution, TSPProblem
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.utils import RL4COTrainer
import torch
from tensordict.tensordict import TensorDict
import random


class TSPRandomSolver(BaseSolver, TSPProblem):
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def solve(self, instance):
        route = list(range(1, instance.num_cities))
        if self.random_seed is not None:
            random.seed(self.random_seed)
        random.shuffle(route)
        route = [0] + route + [0]
        route = torch.tensor(route)
        return TSPSolution(route)


class TSPAttentionModelSolver(BaseRLSolver, TSPProblem):
    def __init__(
        self,
        train_data_size=100000,
        val_data_size=10000,
        test_data_size=10000,
        batch_size=64,
        max_epochs=10,
        accelerator="cpu",
        lr=1e-4,
        num_loc=20,
        num_encoder_layers=1,
    ):
        super().__init__(
            train_data_size,
            val_data_size,
            test_data_size,
            batch_size,
            max_epochs,
            accelerator,
            lr,
        )
        self.save_hyperparameters()
        self.num_encoder_layers = num_encoder_layers
        self.num_loc = num_loc

        self.policy = AttentionModelPolicy(num_encoder_layers=num_encoder_layers)

    def fit(self):
        generator = TSPGenerator(num_loc=self.num_loc)
        env = TSPEnv(generator)
        model = POMO(
            env,
            self.policy,
            batch_size=self.batch_size,
            optimizer_kwargs={"lr": self.lr},
            train_data_size=self.train_data_size,
            val_data_size=self.val_data_size,
            test_data_size=self.test_data_size,
        )
        self.trainer = RL4COTrainer(max_epochs=self.max_epochs, accelerator=self.accelerator)
        self.trainer.fit(model)

    def solve(self, instance):
        locs = instance.locs[None, :, :]
        td = TSPEnv().reset(TensorDict({"locs": locs}), batch_size=[1])
        out = self.policy(td, phase="test", decode_type="greedy", return_actions=True)
        actions = out["actions"].cpu().detach()[0]
        route = torch.concat([actions, actions[[0]]])
        return TSPSolution(route)

    @staticmethod
    def load_model(path):
        return TSPAttentionModelSolver.load_from_checkpoint(path)
