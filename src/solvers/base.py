from abc import ABC, abstractmethod
from src.problems.base import BaseInstance
from src.problems.base import BaseSolution
from src.utils import parse_checkpoint
from pytorch_lightning import LightningModule, Trainer
import os

class BaseSolver(ABC):
    @abstractmethod
    def solve(self, instance: BaseInstance) -> BaseSolution:
        pass


class BaseRLSolver(BaseSolver, LightningModule):
    def __init__(
        self,
        train_data_size=100000,
        val_data_size=10000,
        test_data_size=10000,
        batch_size=16,
        max_epochs=10,
        accelerator="cpu",
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.test_data_size = test_data_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.lr = lr
        self.trainer: Trainer = None

    @abstractmethod
    def fit(self):
        pass

    def save_model(self):
        checkpoint_dir = os.path.join(self.trainer.logger.log_dir, "checkpoints")
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError("No checkpoints found.")
        last_checkpoint = max(
            ckpt_files, key=lambda f: parse_checkpoint(f)
        )
        return os.path.join(checkpoint_dir, last_checkpoint)
    
    @staticmethod
    @abstractmethod
    def load_model(path: str) -> "BaseRLSolver":
        """Loads model given path"""
        pass
