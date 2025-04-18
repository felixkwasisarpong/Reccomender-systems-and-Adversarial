from torch.utils.data import DataLoader
from opacus.data_loader import DPDataLoader
from .NetflixDataset import NetflixDataset
import pytorch_lightning as pl
from typing import Optional
import torch

class NetflixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1024,
        num_workers: int = 8,
        max_samples: Optional[int] = None,
        enable_dp: bool = False,
        test_users_ratio: float = 0.2,
        random_state: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.test_users_ratio = test_users_ratio
        self.random_state = random_state
        self.enable_dp = enable_dp
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Initialize datasets with consistent random state"""
        common_args = {
            'data_dir': self.data_dir,
            'max_samples': self.max_samples,
            'test_users_ratio': self.test_users_ratio,
            'random_state': self.random_state
        }

        if stage == 'fit' or stage is None:
            self.train_dataset = NetflixDataset(mode='train', **common_args)
            self.val_dataset = NetflixDataset(mode='val', **common_args)
        
        if stage == 'test' or stage is None:
            self.test_dataset = NetflixDataset(mode='test', **common_args)

    def train_dataloader(self):
        """Return appropriate dataloader based on DP setting"""
        loader_args = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'persistent_workers': self.num_workers > 0
        }

        if self.enable_dp:
            return DPDataLoader(
                dataset=self.train_dataset,
                sample_rate=self.batch_size / len(self.train_dataset),
                num_workers=self.num_workers,
                pin_memory=True,
                generator=torch.Generator().manual_seed(self.random_state)
            )
        else:
            return DataLoader(
                dataset=self.train_dataset,
                shuffle=True,  # Only shuffle training data
                **loader_args
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Important: validation should not be shuffled
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Important: test should not be shuffled
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def get_num_users(self):
        """Get number of unique users"""
        return self.train_dataset.get_num_users()

    def get_num_items(self):
        """Get number of unique items"""
        return self.train_dataset.get_num_movies()