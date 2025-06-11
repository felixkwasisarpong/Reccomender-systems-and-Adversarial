from torch.utils.data import DataLoader, Dataset
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
        self.member_dataset = None
        self.nonmember_dataset = None

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
            test_path = os.path.join(self.data_dir, "movies.hdf5")
            self.test_dataset = NetflixDataset(
                h5_path=test_path,
                mode='test',
                **common_args
            )

    def setup_mia(self, member_dir: str, nonmember_dir: str):
        """Setup for MIA analysis with member and non-member datasets."""
        self.member_dataset = NetflixDataset(member_dir, mode="train")
        self.nonmember_dataset = NetflixDataset(nonmember_dir, mode="test")

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

    def member_dataloader(self):
        """Dataloader for member dataset."""
        return DataLoader(
            dataset=self.member_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def nonmember_dataloader(self):
        """Dataloader for non-member dataset."""
        return DataLoader(
            dataset=self.nonmember_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_num_users(self):
        return self.train_dataset.get_num_users()

    def get_num_movies(self):
        return self.train_dataset.get_num_movies()
    

    def mia_dataloaders(self, member_dir, nonmember_dir):
        member_ds    = NetflixDataset(member_dir,    mode="train")
        nonmember_ds = NetflixDataset(nonmember_dir, mode="test")

        mia_ds = MIADataset(member_ds, nonmember_ds)
        mia_loader = DataLoader(
            mia_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return [mia_loader]    # list of length 1



class MIADataset(torch.utils.data.Dataset):
    def __init__(self, member_ds, nonmember_ds):
        self.member_ds = member_ds
        self.nonmember_ds = nonmember_ds
        self.len_m = len(member_ds)
        self.len_n = len(nonmember_ds)

    def __len__(self):
        return self.len_m + self.len_n

    def __getitem__(self, idx):
        if idx < self.len_m:
            u, i, r = self.member_ds[idx]
            label = 1
        else:
            u, i, r = self.nonmember_ds[idx - self.len_m]
            label = 0
        return u, i, r, torch.tensor(label, dtype=torch.float32)