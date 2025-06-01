
from typing import Dict, Tuple
from rl4co.models.zoo.l2d import L2DModel
from rl4co.models.zoo.l2d.policy import L2DPolicy
from rl4co.envs import FJSPEnv
from rl4co.utils.trainer import RL4COTrainer
import torch


from src.problems.jsp.fjsp import FJSP, FJSPInstance, FJSPSolution
from src.solvers.base import BaseRLSolver


class FJSPL2DSolver(BaseRLSolver, FJSP):
    def __init__(
        self,
        train_data_size: int = 100000,
        val_data_size: int = 10000,
        test_data_size: int = 10000,
        batch_size: int = 64,
        max_epochs: int = 10,
        accelerator: str = "cpu",
        lr: float = 1e-4,
        num_encoder_layers: int = 1,
        embed_dim: int = 32,
        num_heads: int = 2,
        feedforward_hidden: int = 64,
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feedforward_hidden = feedforward_hidden

        self.policy = L2DPolicy(embed_dim=embed_dim, num_encoder_layers=num_encoder_layers, env_name="fjsp")

    def fit(self, fjsp_env_params: Dict):


        env = FJSPEnv(**fjsp_env_params)

        model = L2DModel(env,
                 policy=self.policy, 
                 baseline="rollout",
                 batch_size=self.batch_size,
                 train_data_size=self.train_data_size,
                 val_data_size=1_000,
                 optimizer_kwargs={"lr": 1e-4})
        

        self.trainer = RL4COTrainer(
            max_epochs=self.max_epochs, accelerator=self.accelerator
        )

        self.trainer.fit(model)


    def solve(self, instance: FJSPInstance) -> FJSPSolution:
        
        env =  FJSPEnv()
        td = env.reset(instance.td_init)
        actions_sequence = []

        (op_emb, ma_emb), init = self.policy.encoder(td)

        while not td["done"].all():
            action, job, machine = self.get_action(td, (op_emb, ma_emb))
            td["action"] = torch.Tensor([action]).to(int)
            td = env.step(td)["next"]
            actions_sequence.append(action)

        return FJSPSolution(actions_sequence=actions_sequence)
    

    def get_action(self, td, features) -> Tuple[int, int, int]:

        n_machines = td["proc_times"].shape[1]

        logits, mask = self.policy.decoder(td, features, num_starts=0)
        action = logits.masked_fill(~mask, -torch.inf).argmax(1)

        job = (action - 1) // n_machines
        machine = (action - 1) % n_machines

        return action, job, machine

    @staticmethod
    def load_model(path):
        return FJSPL2DSolver.load_from_checkpoint(path)
