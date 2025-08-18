import os
from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from .NetflixDataset import NetflixDataset
from torch.utils.data import Dataset
try:
    from opacus import PrivacyEngine
    from opacus.utils.uniform_sampler import UniformWithReplacementSampler as DPDataLoader
except ImportError:
    DPDataLoader = None

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Optional
import os
import numpy as np


class ImprovedMIADataset(Dataset):
    """
    Improved MIA Dataset using the 1M dataset as true non-members.
    This creates a more realistic scenario where:
    - Members: Users from 100k dataset used in training
    - Non-members: Users from 1M dataset (completely unseen during training)
    """
    
    def __init__(self, member_dataset, nonmember_full_dataset, balance_ratio=1.0, sample_size=None):
        """
        Args:
            member_dataset: Training dataset (100k subset - known members)
            nonmember_full_dataset: Full 1M dataset (true non-members)
            balance_ratio: Ratio of non-members to members
            sample_size: Maximum number of samples to use (None = use all)
        """
        self.member_dataset = member_dataset
        self.nonmember_full_dataset = nonmember_full_dataset
        self.balance_ratio = balance_ratio
        self._offset_nonmember_ids = 0  # used when datasets reindex independently
        self._sample_keys_logged = False
        # Get unique users from each dataset
        self.member_users = self._get_unique_users(member_dataset, role="member")
        self.nonmember_users = self._get_unique_users(nonmember_full_dataset, role="nonmember")
        # Remove any overlap (users that appear in both datasets)
        self.true_nonmember_users = self.nonmember_users - self.member_users
        print(f"Member users: {len(self.member_users)}")
        print(f"Non-member users (total): {len(self.nonmember_users)}")
        print(f"True non-member users (after removing overlap): {len(self.true_nonmember_users)}")
        if len(self.true_nonmember_users) == 0:
            print("[MIA][WARNING] No true non-member users found. This usually means both datasets reindex users independently.\n"
                  "Attempting fallback: offsetting non-member user IDs to force disjoint sets.\n"
                  "Better fix: expose a stable raw user id (e.g., 'raw_user_id') in NetflixDataset.")
            self._offset_nonmember_ids = 10_000_000
            # Recompute sets with offset applied to nonmembers
            self.nonmember_users = self._get_unique_users(self.nonmember_full_dataset, role="nonmember")
            self.true_nonmember_users = self.nonmember_users - self.member_users
            print(f"[MIA][FALLBACK] True non-member users (offset mode): {len(self.true_nonmember_users)}")
        # Create samples
        self.samples = self._create_balanced_samples(sample_size)
        print(f"Final MIA dataset: {len(self.samples)} samples")
        
    def _get_unique_users(self, dataset, role: str = "member"):
        """Extract unique user IDs, preferring a stable/global ID if present.
        If only reindexed 'user_id' exists, we optionally offset nonmember IDs to avoid false overlaps.
        """
        preferred_keys = (
            'raw_user_id', 'original_user_id', 'global_user_id', 'uid', 'user_orig', 'user'
        )
        users = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            if not self._sample_keys_logged:
                try:
                    print("[MIA] Sample keys available:", list(sample.keys()))
                except Exception:
                    pass
                self._sample_keys_logged = True
            # Prefer raw/global user id fields if available
            uid_val = None
            for k in preferred_keys:
                if k in sample:
                    v = sample[k]
                    uid_val = int(v.item() if hasattr(v, 'item') else int(v))
                    break
            if uid_val is None:
                v = sample['user_id']
                uid_val = int(v.item() if hasattr(v, 'item') else int(v))
            # If datasets are reindexed independently, allow an offset for nonmembers
            if role == "nonmember" and self._offset_nonmember_ids:
                uid_val = uid_val + int(self._offset_nonmember_ids)
            users.add(uid_val)
        if users:
            umin, umax = min(users), max(users)
            print(f"[MIA] Unique {role} users: {len(users)} (min={umin}, max={umax})")
        return users
    
    def _create_balanced_samples(self, sample_size):
        """Create balanced member/non-member samples"""
        # Sample member interactions
        member_samples = []
        target_members = len(self.member_dataset) if sample_size is None else sample_size // 2
        if target_members > len(self.member_dataset):
            target_members = len(self.member_dataset)
        member_indices = np.random.choice(len(self.member_dataset), target_members, replace=False)
        for idx in member_indices:
            sample = self.member_dataset[idx]
            sample_with_label = {**sample, 'label': torch.tensor(1, dtype=torch.long)}  # Member
            member_samples.append(sample_with_label)
        # Sample non-member interactions from users not in training
        nonmember_samples = []
        target_nonmembers = int(len(member_samples) * self.balance_ratio)
        # Collect all interactions from true non-member users
        true_nonmember_interactions = []
        for i in range(len(self.nonmember_full_dataset)):
            sample = self.nonmember_full_dataset[i]
            uid = int(sample['user_id'].item())
            if self._offset_nonmember_ids:
                uid = uid + int(self._offset_nonmember_ids)
            if uid in self.true_nonmember_users:
                true_nonmember_interactions.append(sample)
        if len(true_nonmember_interactions) < target_nonmembers:
            print(f"Warning: Only {len(true_nonmember_interactions)} non-member interactions available, "
                  f"requested {target_nonmembers}")
            target_nonmembers = len(true_nonmember_interactions)
        # Randomly sample non-member interactions
        if target_nonmembers > 0:
            nonmember_indices = np.random.choice(
                len(true_nonmember_interactions),
                target_nonmembers,
                replace=False
            )
            for idx in nonmember_indices:
                sample = true_nonmember_interactions[idx]
                sample_with_label = {**sample, 'label': torch.tensor(0, dtype=torch.long)}  # Non-member
                nonmember_samples.append(sample_with_label)
        # Combine and shuffle
        all_samples = member_samples + nonmember_samples
        np.random.shuffle(all_samples)
        print(f"Created {len(member_samples)} member samples and {len(nonmember_samples)} non-member samples")
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class UserLevelMIADataset(Dataset):
    """
    User-level MIA dataset that aggregates interactions per user.
    This is more aligned with user privacy concerns.
    """
    
    def __init__(self, member_dataset, nonmember_full_dataset, interactions_per_user=5):
        """
        Args:
            member_dataset: Training dataset (members)
            nonmember_full_dataset: Full dataset (for non-members)
            interactions_per_user: Number of interactions to sample per user
        """
        self.interactions_per_user = interactions_per_user
        
        # Group interactions by user
        member_user_interactions = self._group_by_user(member_dataset)
        nonmember_user_interactions = self._group_by_user(nonmember_full_dataset)
        
        # Get true non-member users (not in training)
        member_users = set(member_user_interactions.keys())
        true_nonmember_users = {
            user: interactions for user, interactions in nonmember_user_interactions.items()
            if user not in member_users
        }
        
        print(f"Member users: {len(member_user_interactions)}")
        print(f"True non-member users: {len(true_nonmember_users)}")
        
        # Create user-level samples
        self.samples = self._create_user_samples(member_user_interactions, true_nonmember_users)
        
        print(f"User-level MIA dataset: {len(self.samples)} user samples")
    
    def _group_by_user(self, dataset):
        """Group dataset interactions by user ID"""
        user_interactions = {}
        for i in range(len(dataset)):
            sample = dataset[i]
            user_id = sample['user_id'].item()
            if user_id not in user_interactions:
                user_interactions[user_id] = []
            user_interactions[user_id].append(sample)
        return user_interactions
    
    def _create_user_samples(self, member_users, nonmember_users):
        """Create samples where each sample represents a user with multiple interactions"""
        samples = []
        
        # Sample from member users
        for user_id, interactions in member_users.items():
            if len(interactions) >= self.interactions_per_user:
                sampled_interactions = np.random.choice(
                    interactions, self.interactions_per_user, replace=False
                ).tolist()
            else:
                sampled_interactions = interactions
            
            samples.append({
                'user_id': user_id,
                'interactions': sampled_interactions,
                'label': torch.tensor(1, dtype=torch.long)  # Member
            })
        
        # Sample from non-member users (balance the dataset)
        target_nonmembers = len(member_users)
        nonmember_user_list = list(nonmember_users.keys())
        
        if len(nonmember_user_list) < target_nonmembers:
            target_nonmembers = len(nonmember_user_list)
        
        selected_nonmember_users = np.random.choice(
            nonmember_user_list, target_nonmembers, replace=False
        )
        
        for user_id in selected_nonmember_users:
            interactions = nonmember_users[user_id]
            if len(interactions) >= self.interactions_per_user:
                sampled_interactions = np.random.choice(
                    interactions, self.interactions_per_user, replace=False
                ).tolist()
            else:
                sampled_interactions = interactions
            
            samples.append({
                'user_id': user_id,
                'interactions': sampled_interactions,
                'label': torch.tensor(0, dtype=torch.long)  # Non-member
            })
        
        np.random.shuffle(samples)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_sample = self.samples[idx]
        # For simplicity, return the first interaction (you could aggregate or sample)
        interaction = user_sample['interactions'][0]
        return {**interaction, 'label': user_sample['label']}


class NetflixDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 1024, num_workers: int = 4, 
                 max_samples: Optional[int] = None, pred_file: Optional[str] = "dp_mid",        
                 train_split_ratio: float = 0.8, val_split_ratio: float = 0.1, 
                 random_state: int = 42, return_attrs: bool = True, enable_dp: bool = False,
                 mia_strategy: str = "improved"):  # New parameter for MIA strategy
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.random_state = random_state
        self.return_attrs = return_attrs
        self.enable_dp = enable_dp
        self.pred_file = pred_file
        self.mia_strategy = mia_strategy  # "improved", "user_level", or "original"

    def _loader_kwargs(self, drop_last: bool = False):
        """Build DataLoader kwargs that are safe for CPU/MPS.
        - prefetch_factor is only set when num_workers > 0 (PyTorch requirement)
        - persistent_workers only when num_workers > 0
        - pin_memory False by default for MPS/CPU
        """
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=drop_last,
        )
        if self.num_workers and self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4
        return kwargs

    def setup(self, stage: Optional[str] = None):
        # Load 100k dataset for training
        full_dataset = NetflixDataset(
            h5_path=os.path.join(self.data_dir, "movielens_100k_with_attrs.hdf5"),
            max_samples=self.max_samples,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode='full',
        )

        # Compute lengths
        total_len = len(full_dataset)
        train_len = int(self.train_split_ratio * total_len)
        val_len = int(self.val_split_ratio * total_len)
        test_len = total_len - train_len - val_len

        # Split dataset deterministically
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_state)
        )
        self.num_train = len(self.train_dataset)
        
        # Load 1M dataset (used for prediction and as true non-members for MIA)
        self.predict_dataset = NetflixDataset(
            h5_path=os.path.join(self.data_dir, "movielens_1m_with_attrs.hdf5"),
            max_samples=None,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode='predict'
        )
        
        # Load full 1M dataset for MIA (different from predict_dataset)
        self.full_1m_dataset = NetflixDataset(
            h5_path=os.path.join(self.data_dir, "movielens_1m_with_attrs.hdf5"),
            max_samples=None,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode='full'  # Load as full dataset, not predict mode
        )

        if stage == "attack":
            attack_full_path = os.path.join(self.data_dir,  f"{self.pred_file}.hdf5")
            attack_dataset = NetflixDataset(
                h5_path=attack_full_path,
                max_samples=self.max_samples,
                random_state=self.random_state,
                return_attrs=self.return_attrs,
                mode='full'
            )
            self.attack_dataset = WGANInputDataset(attack_dataset)

        if stage == "classifier_attack":
            full = NetflixDataset(
                h5_path=os.path.join(self.data_dir, f"{self.pred_file}.hdf5"),
                max_samples=self.max_samples,
                random_state=self.random_state,
                return_attrs=self.return_attrs,
                mode='full'
            )

            total_len = len(full)
            train_len = int(self.train_split_ratio * total_len)
            val_len = int(self.val_split_ratio * total_len)
            test_len = total_len - train_len - val_len

            self.class_train, self.class_val, self.class_test = random_split(
                full, [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(self.random_state)
            )

    def train_dataloader(self):
        # Use plain loaders; the model will wrap/replace this in opacus mode.
        kwargs = self._loader_kwargs(drop_last=True)
        kwargs["shuffle"] = True
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=True)
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        kwargs["shuffle"] = True
        return DataLoader(self.test_dataset, **kwargs)

    def predict_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        return DataLoader(self.predict_dataset, **kwargs)

    def class_attack_train_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        kwargs["shuffle"] = True
        return DataLoader(self.class_train, **kwargs)

    def class_attack_val_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        return DataLoader(self.class_val, **kwargs)

    def class_attack_test_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        return DataLoader(self.class_test, **kwargs)

    def attack_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        kwargs["shuffle"] = True
        return DataLoader(self.attack_dataset, **kwargs)

    def get_num_users(self):
        return self.train_dataset.dataset.get_num_users()

    def get_num_movies(self):
        return self.train_dataset.dataset.get_num_movies()

    def mia_dataloaders(self):
        """Create MIA dataloaders using different strategies"""
        if self.mia_strategy == "improved":
            # Use 1M dataset as true non-members
            mia_ds = ImprovedMIADataset(
                member_dataset=self.train_dataset,
                nonmember_full_dataset=self.full_1m_dataset,
                balance_ratio=1.0,
                sample_size=20000  # Limit sample size for efficiency
            )
        elif self.mia_strategy == "user_level":
            # User-level MIA approach
            mia_ds = UserLevelMIADataset(
                member_dataset=self.train_dataset,
                nonmember_full_dataset=self.full_1m_dataset,
                interactions_per_user=3
            )
        else:  # "original"
            # Original approach (train vs test from same dataset)
            mia_ds = MIADataset(self.train_dataset, self.test_dataset)

        kwargs = self._loader_kwargs(drop_last=False)
        kwargs["shuffle"] = False
        loader = DataLoader(mia_ds, **kwargs)
        print(f"[INFO] MIA loader ({self.mia_strategy}): {len(mia_ds)} samples")
        return [loader]


# Keep original classes for backward compatibility
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
            data = self.member_ds[idx]
            label = 1
        else:
            data = self.nonmember_ds[idx - self.len_m]
            label = 0

        example = {
            'user_id': data['user_id'],
            'item_id': data['item_id'],
            'rating': data['rating'],
            'label': torch.tensor(label, dtype=torch.long)
        }

        for key in ['gender', 'age', 'occupation', 'genre']:
            if key in data:
                example[key] = data[key]

        return example


class WGANInputDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        # Normalize user_id and item_id according to ML-100K dataset ranges
        user_id    = item['user_id'].float().unsqueeze(0) / 943
        item_id    = item['item_id'].float().unsqueeze(0) / 1682
        gender     = item['gender'].float().unsqueeze(0)
        age        = item['age'].unsqueeze(0) / 100
        occupation = item['occupation'].float().unsqueeze(0) / 20
        genre      = item['genre']
        rating     = item['rating'].unsqueeze(0) / 5.0

        full_vec = torch.cat([
            user_id, item_id, gender, age, occupation, genre, rating
        ], dim=0)

        return full_vec


class AttributeInferenceDataset(Dataset):
    def __init__(self, base_dataset, target_attr="gender"):
        self.base_dataset = base_dataset
        self.target_attr = target_attr
        self.attr_keys = ["gender", "occupation", "age"]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        user_id    = item['user_id'].float().unsqueeze(0) / 6000
        item_id    = item['item_id'].float().unsqueeze(0) / 4000
        rating     = item['rating'].float().unsqueeze(0) / 5.0
        age        = item['age'].float().unsqueeze(0) / 100
        occupation = item['occupation'].float().unsqueeze(0) / 20
        gender     = item['gender'].float().unsqueeze(0)
        genre      = item['genre']

        feature = torch.cat([user_id, item_id, rating, age, occupation, genre], dim=0)

        if self.target_attr == "age":
            label = item["age"].float()
        elif self.target_attr == "gender":
            label = item["gender"].long()
        elif self.target_attr == "occupation":
            label = item["occupation"].long()
        else:
            raise ValueError(f"Unsupported target_attr: {self.target_attr}")

        return feature, label