from torch.utils.data import DataLoader
from opacus.data_loader import DPDataLoader
from .NetflixDataset import NetflixDataset
import pytorch_lightning as pl
from typing import Optional
import os
import torch

class NetflixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1024,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
        enable_dp: bool = False,
        train_split_ratio: float = 0.9,
        random_state: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.enable_dp = enable_dp
        self.train_split_ratio = train_split_ratio
        self.random_state = random_state
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        common_args = {
            'max_samples': self.max_samples,
            'random_state': self.random_state
        }

        if stage == 'fit' or stage is None:
            train_path = os.path.join(self.data_dir, "netflix_data.hdf5")
            self.train_dataset = NetflixDataset(
                h5_path=train_path,
                mode='train',
                split_ratio=self.train_split_ratio,
                **common_args
            )
            self.val_dataset = NetflixDataset(
                h5_path=train_path,  # Same file for validation split
                mode='val',
                split_ratio=self.train_split_ratio,
                **common_args
            )
        
        if stage == 'test' or stage is None:
            test_path = os.path.join(self.data_dir, "test_data.hdf5")
            self.test_dataset = NetflixDataset(
                h5_path=test_path,
                mode='test',
                **common_args
            )

    def train_dataloader(self):
        loader_args = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'persistent_workers': self.num_workers > 0
        }

        if self.enable_dp:
            return DPDataLoader(
                dataset=self.train_dataset,
                sample_rate=self.batch_size/len(self.train_dataset),
                num_workers=self.num_workers,
                pin_memory=True,
                generator=torch.Generator().manual_seed(self.random_state))
        else:
            return DataLoader(
                dataset=self.train_dataset,
                shuffle=True,
                **loader_args)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size*2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size*2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)

    def get_num_users(self):
        return self.train_dataset.get_num_users()

    def get_num_movies(self):
        return self.train_dataset.get_num_movies()