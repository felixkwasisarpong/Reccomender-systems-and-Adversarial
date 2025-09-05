import os
import h5py
from typing import Optional
import torch
import torch.nn.functional as F
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

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
            'raw_user_id', 'original_user_id', 'global_user_id', 'uid', 'user_orig', 'user', 'user_id'
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
            # Prefer raw/global user id fields if available, including 'user_id' at the end
            uid_val = None
            for k in preferred_keys:
                if k in sample:
                    v = sample[k]
                    uid_val = int(v.item() if hasattr(v, 'item') else int(v))
                    break
            if uid_val is None:
                # Should not happen, but fallback
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
        # Use the same preferred_keys as in _get_unique_users
        preferred_keys = (
            'raw_user_id', 'original_user_id', 'global_user_id', 'uid', 'user_orig', 'user', 'user_id'
        )
        for i in range(len(self.nonmember_full_dataset)):
            sample = self.nonmember_full_dataset[i]
            uid = None
            for k in preferred_keys:
                if k in sample:
                    v = sample[k]
                    uid = int(v.item() if hasattr(v, 'item') else int(v))
                    break
            if uid is None:
                v = sample['user_id']
                uid = int(v.item() if hasattr(v, 'item') else int(v))
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
                 max_samples: Optional[int] = None, pred_file: Optional[str] = "ml1m_all_train",        
                 train_split_ratio: float = 0.8, val_split_ratio: float = 0.1, 
                 random_state: int = 42, return_attrs: bool = True, enable_dp: bool = False,
                 mia_strategy: str = "improved", mia_nonmember_path: Optional[str] = None):  # New parameter for MIA strategy
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
        # Optional explicit path to non-member pool (full 1M) for MIA; if None, falls back to data_dir/pred_file.hdf5
        self.mia_nonmember_path = mia_nonmember_path

    def _build_netflix_base_from_hdf5(self, path, dataset: str | None = None):
        """Return a base NetflixDataset from an HDF5 file path, with sanity checks."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[classifier_attack] HDF5 not found: {path}")
        # Optional: sanity check keys present
        try:
            with h5py.File(path, "r") as fchk:
                _keys = list(fchk.keys())
                # Store last-inspected keys for debugging
                self._last_hdf5_keys = _keys
        except Exception:
            pass
        return NetflixDataset(
            h5_path=path,
            max_samples=self.max_samples,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode="full",
        )

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
            #h5_path=os.path.join(self.data_dir, "ml1m_all_train.hdf5"),
            h5_path=os.path.join(self.data_dir, f"{self.pred_file}.hdf5"),
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
            h5_path=os.path.join(self.data_dir, f"{self.pred_file}.hdf5"),
            #h5_path=os.path.join(self.data_dir, "ml1m_all_holdout.hdf5"),
            max_samples=None,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode='predict'
        )
        
        # Load full 1M dataset for MIA (different from predict_dataset)
        # Prefer explicit mia_nonmember_path if provided; else use data_dir/pred_file.hdf5
        nm_path = self.mia_nonmember_path
        if not nm_path:
            nm_path = os.path.join(self.data_dir, f"{self.pred_file}.hdf5")
            #nm_path = os.path.join(self.data_dir,"ml1m_all_holdout.hdf5")
        print(f"[MIA] Using non-member HDF5: {nm_path}")
        self.full_1m_dataset = NetflixDataset(
            h5_path=nm_path,
            max_samples=None,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode='full'  # Load as full dataset, not predict mode
        )

        if stage == "attack":
            attack_full_path = os.path.join(self.data_dir, f"{self.pred_file}.hdf5")

            attack_base = NetflixDataset(
                h5_path=attack_full_path,
                max_samples=self.max_samples,
                random_state=self.random_state,
                return_attrs=self.return_attrs,
                mode='full'
            )

            attack_01 = WGANInputDataset(
                attack_base,
                use_onehot_occ=True,    # <- was False
                use_bucket_age=True,
                one_based_ids=True,
            )

            # Was: _ToPM1 defined locally -> pickling error
            self.attack_dataset = ToPM1Dataset(attack_01)

            # Optional handle for plotting/denorm helpers (set before deriving dims)
            self.attack_dataset_01 = attack_01

            # --- Derive WGAN layout dims from dataset ---
            try:
                # Preferred: from adapter (handles optional genres / occ one-hot/scalar)
                self.expected_dim = int(attack_01.expected_dim)
            except Exception:
                # Fallback probe from [-1,1] wrapped dataset
                x0 = self.attack_dataset[0]
                self.expected_dim = int(x0.numel())
            self.rating_idx = int(self.expected_dim - 1)
            print(f"[DM][WGAN] expected_dim={self.expected_dim} rating_idx={self.rating_idx}")

        if stage == "classifier_attack":
            train_source = getattr(self, "attack_train_source", "synthetic_outputs")
            test_source  = getattr(self, "attack_test_source",  "predicted_data")

            synthetic_path = getattr(self, "synthetic_hdf5_path", None)
            if not synthetic_path:
                synthetic_path = os.path.join(self.data_dir, f"{self.pred_file}.hdf5")
            blackbox_path = getattr(self, "blackbox_hdf5_path", None)
            if not blackbox_path:
                bb_dir = getattr(self, "blackbox_dir", None)
                if bb_dir:
                    blackbox_path = os.path.join(bb_dir, f"{self.pred_file}.hdf5")
                else:
                    base_dir = self.data_dir.rstrip("/\\")
                    cand = os.path.join(os.path.dirname(base_dir), "predicted_data", f"{self.pred_file}.hdf5")
                    blackbox_path = cand if os.path.isfile(cand) else os.path.join(self.data_dir, f"{self.pred_file}.hdf5")

            # Build base datasets
            if train_source == "synthetic_outputs":
                base_train = self._build_netflix_base_from_hdf5(synthetic_path, dataset="synthetic")
            elif train_source == "predicted_data":
                base_train = self._build_netflix_base_from_hdf5(blackbox_path, dataset="predictions")
            else:
                raise ValueError(f"Unknown attack_train_source: {train_source}")

            if test_source == "predicted_data":
                base_test = self._build_netflix_base_from_hdf5(blackbox_path, dataset="predictions")
            elif test_source == "synthetic_outputs":
                base_test = self._build_netflix_base_from_hdf5(synthetic_path, dataset="synthetic")
            else:
                raise ValueError(f"Unknown attack_test_source: {test_source}")

            # Attack knobs
            target_attr         = getattr(self, "attack_target", "gender")
            include_identifiers = bool(getattr(self, "attack_include_identifiers", False))
            include_rating      = bool(getattr(self, "attack_include_rating", True))

            # Wrap datasets
            full_train = AttributeInferenceDataset(
                base_train,
                target_attr=target_attr,
                include_identifiers=include_identifiers,
                include_rating=include_rating,
                num_users=getattr(self, "num_users", None),
                num_items=getattr(self, "num_items", None),
                identifiers_one_based=True,
                remap_occ_1_based=getattr(self, "attack_remap_occ_1_based", False),
                age_label_mode=getattr(self, "attack_age_label_mode", "code"),
            )
            full_test = AttributeInferenceDataset(
                base_test,
                target_attr=target_attr,
                include_identifiers=include_identifiers,
                include_rating=include_rating,
                num_users=getattr(self, "num_users", None),
                num_items=getattr(self, "num_items", None),
                identifiers_one_based=True,
                remap_occ_1_based=getattr(self, "attack_remap_occ_1_based", False),
                age_label_mode=getattr(self, "attack_age_label_mode", "code"),
            )

            # Split TRAIN into train/val
            N = len(full_train)
            idx = np.arange(N)
            rng = np.random.default_rng(getattr(self, "random_state", 42))
            rng.shuffle(idx)
            tr_cut = int(0.9 * N)
            train_idx, val_idx = idx[:tr_cut], idx[tr_cut:]

            self.class_train = Subset(full_train, train_idx.tolist())
            self.class_val   = Subset(full_train, val_idx.tolist())
            self.class_test  = full_test


    def class_attack_train_dataloader(self):
        kw = self._loader_kwargs(drop_last=False); kw["shuffle"] = True
        return DataLoader(self.class_train, **kw)

    def class_attack_val_dataloader(self):
        kw = self._loader_kwargs(drop_last=False); kw["shuffle"] = False
        return DataLoader(self.class_val, **kw)

    def class_attack_test_dataloader(self):
        kw = self._loader_kwargs(drop_last=False); kw["shuffle"] = False
        return DataLoader(self.class_test, **kw)

    def train_dataloader(self):
        # Use plain loaders; the model will wrap/replace this in opacus mode.
        kwargs = self._loader_kwargs(drop_last=True)
        kwargs["shuffle"] = True
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=True)
        kwargs["shuffle"] = False
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        kwargs["shuffle"] = False
        return DataLoader(self.test_dataset, **kwargs)

    def predict_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        return DataLoader(self.predict_dataset, **kwargs)


    def get_wgan_expected_dim(self) -> Optional[int]:
        return getattr(self, "expected_dim", None)

    def get_wgan_rating_idx(self) -> Optional[int]:
        return getattr(self, "rating_idx", None)

    def attack_dataloader(self):
        kwargs = self._loader_kwargs(drop_last=False)
        kwargs["shuffle"] = True
        return DataLoader(self.attack_dataset, **kwargs)


    @torch.no_grad()
    def _to_original_units(self, x01: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "attack_dataset_01"):
            raise RuntimeError("attack_dataset_01 not available; call setup('attack') first.")
        ds = self.attack_dataset_01  # WGANInputDataset instance

        x = x01.clone()
        # Order: [uid, iid, age, gender, occ, (genre[K]), rating]
        age_idx = 2
        gender_idx = 3
        occ_start = 4
        occ_dim = ds.OCC_CLASSES if ds.use_onehot_occ else 1
        genre_start = occ_start + occ_dim
        genre_dim = ds.genre_dim if getattr(ds, 'has_genre', False) else 0
        rating_idx = genre_start + genre_dim

        # Denormalize
        x[:, gender_idx] = x[:, gender_idx].round().clamp_(0, 1)
        if not ds.use_onehot_occ:
            x[:, occ_start] = (x[:, occ_start] * 20.0).round().clamp_(0, 20)
        x[:, age_idx] = x[:, age_idx] * 100.0
        mul = float(getattr(ds, 'rating_divisor', 5.0))
        x[:, rating_idx] = x[:, rating_idx] * mul
        return x

    @torch.no_grad()
    def sample_real_original_units(self, k: int = 100000) -> torch.Tensor:
        """Sample up to k rows from the [0,1] attack view and return in original units."""
        if not hasattr(self, "attack_dataset_01"):
            raise RuntimeError("attack_dataset_01 not available; call setup('attack') first.")
        m = len(self.attack_dataset_01)
        idx = torch.randperm(m)[: min(m, k)]
        rows = [self.attack_dataset_01[int(i)] for i in idx]
        x01 = torch.stack(rows, dim=0)
        return self._to_original_units(x01)
    
    
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
    

# ---- put this at top-level (e.g., near your other Dataset classes) ----
class ToPM1Dataset(torch.utils.data.Dataset):
    """Wraps a [0,1]-scaled dataset and returns [-1,1]-scaled tensors."""
    def __init__(self, base_ds):
        self.base_ds = base_ds
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        x01 = self.base_ds[idx]          # tensor [25] in [0,1]
        return x01 * 2.0 - 1.0           # -> [-1,1]


class WGANInputDataset(torch.utils.data.Dataset):
    """
    Emits a MovieLens-style vector in [0,1] with dynamic layout:

        [ user_id, item_id, age, gender, occupation(s), (optional) genre[K], rating ]

    - IDs are scaled to [0,1] using detected num_users/num_items (or overrides).
    - Age: scalar in years → age/100 (or bucketed if use_bucket_age=True).
    - Gender: assumed 0/1 already.
    - Occupation: either scalar (/20) or one-hot(21).
    - Genre: included iff base sample has 'genre' (shape [K]).
    - Rating: prefers 'pred_rating' (from predicted HDF5) if present; else 'rating'.
              Auto-normalizes: if value ∈ [0,1] keep; else divide by 5.0.

    The final dimension is computed dynamically and asserted.
    """
    def __init__(
        self,
        base_dataset,
        use_onehot_occ: bool = False,
        use_bucket_age: bool = False,
        one_based_ids: bool = True,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
    ):
        import torch.nn.functional as F

        self.base_dataset = base_dataset
        self.use_onehot_occ = use_onehot_occ
        self.use_bucket_age = use_bucket_age   # False -> continuous age/100
        self.one_based_ids = one_based_ids

        # Try to detect sizes from the wrapped NetflixDataset
        detected_users = getattr(base_dataset, 'num_users', None)
        detected_items = getattr(base_dataset, 'num_items', None)

        if (detected_users is None or detected_items is None) and hasattr(base_dataset, 'get_vocab_sizes'):
            try:
                vs = base_dataset.get_vocab_sizes()
                if detected_users is None and 'num_users' in vs:
                    detected_users = int(vs['num_users'])
                if detected_items is None and 'num_items' in vs:
                    detected_items = int(vs['num_items'])
            except Exception:
                pass

        # Explicit overrides win
        if num_users is not None:
            detected_users = int(num_users)
        if num_items is not None:
            detected_items = int(num_items)

        # Safe ML-1M fallbacks if nothing was detected
        if detected_users is None:
            detected_users = 6038
        if detected_items is None:
            detected_items = 3952

        self.num_users = int(detected_users)
        self.num_items = int(detected_items)

        # Denominators for 0..(N-1) range
        self.u_den = max(1, self.num_users - 1)
        self.i_den = max(1, self.num_items - 1)

        self.OCC_CLASSES = 21
        self._age_codes = torch.tensor([1., 18., 25., 35., 45., 50., 56.], dtype=torch.float32)

        # Peek one sample to discover schema
        probe = self.base_dataset[0]
        self.has_genre = 'genre' in probe
        self.genre_dim = int(probe['genre'].numel()) if self.has_genre else 0

        # Which rating key to use
        self.rating_key = 'pred_rating' if 'pred_rating' in probe else 'rating'

        # Decide rating normalization: if values look like 0..1 keep; else divide by 5
        r_val = float(probe[self.rating_key])
        self.rating_divisor = 1.0 if 0.0 <= r_val <= 1.0 else 5.0

        # Compute expected dim dynamically
        occ_dim = self.OCC_CLASSES if self.use_onehot_occ else 1
        self.expected_dim = 2 + 1 + 1 + occ_dim + (self.genre_dim if self.has_genre else 0) + 1
        # IDs(2) + age(1) + gender(1) + occ + genre[K] + rating(1)

        self._logged_once = False

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        import torch
        import torch.nn.functional as F

        it = self.base_dataset[idx]

        # IDs: convert 1..N -> 0..N-1 if one_based_ids=True, then scale to [0,1]
        u = it['user_id'].float() - (1.0 if self.one_based_ids else 0.0)
        i = it['item_id'].float() - (1.0 if self.one_based_ids else 0.0)
        user_id = (u.unsqueeze(0) / float(self.u_den)).clamp_(0, 1)
        item_id = (i.unsqueeze(0) / float(self.i_den)).clamp_(0, 1)

        # Age
        if self.use_bucket_age:
            codes = self._age_codes.to(it['age'].device)
            idx_code = torch.argmin(torch.abs(codes - it['age']))
            age = (idx_code.float().unsqueeze(0) / 6.0).clamp_(0, 1)  # 0..6 -> /6
        else:
            age = (it['age'].float().unsqueeze(0) / 100.0).clamp_(0, 1)  # years/100

        # Gender (already 0/1)
        gender = it['gender'].float().unsqueeze(0)

        # Occupation
        if self.use_onehot_occ:
            occupation = F.one_hot(it['occupation'].long(), num_classes=self.OCC_CLASSES).float()
        else:
            occupation = (it['occupation'].float().unsqueeze(0) / 20.0).clamp_(0, 1)

        blocks = [user_id, item_id, age, gender, occupation]

        # Genres (optional)
        if self.has_genre:
            genre = it['genre'].float()
            # Ensure correct shape [K]
            if genre.dim() == 1:
                pass
            elif genre.dim() == 2 and genre.shape[0] == 1:
                genre = genre.squeeze(0)
            else:
                genre = genre.view(-1)
            blocks.append(genre)

        # Rating (prefer predicted if present), normalize to [0,1] if needed
        rating_raw = it[self.rating_key].float().unsqueeze(0)
        rating = (rating_raw / float(self.rating_divisor)).clamp_(0, 1)
        blocks.append(rating)

        x = torch.cat(blocks, dim=0).to(torch.float32)

        if not self._logged_once:
            self._logged_once = True
            print(f"[WGANInputDataset] schema: has_genre={self.has_genre}, genre_dim={self.genre_dim}, "
                  f"rating_key='{self.rating_key}', rating_div={self.rating_divisor}")
            print(f"[WGANInputDataset] sample shape={tuple(x.shape)}, min={float(x.min()):.4f}, "
                  f"max={float(x.max()):.4f}, mean={float(x.mean()):.4f}, std={float(x.std()):.4f}")

        if x.shape[0] != self.expected_dim:
            raise RuntimeError(
                f"WGANInputDataset expected {self.expected_dim}-D "
                f"(has_genre={self.has_genre}, use_onehot_occ={self.use_onehot_occ}, genre_dim={self.genre_dim}), "
                f"got {tuple(x.shape)}"
            )

        if not torch.isfinite(x).all():
            raise RuntimeError("WGANInputDataset produced non-finite values")

        if x.min() < 0.0 or x.max() > 1.0:
            raise RuntimeError("WGANInputDataset produced values outside [0,1]")

        return x



class AttributeInferenceDataset(Dataset):
    """
    Builds features for attribute inference while *excluding* the target attribute from X.
    Scales inputs to the same ranges used by your WGAN / synthetic canonical 25-D view:
      - ids (optional): user_id/(U-1), item_id/(I-1)
      - rating (optional): if 'pred_rating' or 'rating' given:
            * if in [0,1] -> keep, else divide by 5
      - age: /100  (excluded if target_attr == "age")
      - occupation: /20  (excluded if target_attr == "occupation")
      - gender: {0,1} (excluded if target_attr == "gender")
      - genre: multi-hot [G]; left as 0/1

    Accepts items coming from either:
      - synthetic 25-D view (we expect dict fields already present), or
      - structured 'predictions' (with 'pred_rating', 'genre', etc.)
    """
    def __init__(
        self,
        base_dataset,
        target_attr: str = "gender",
        include_identifiers: bool = False,
        include_rating: bool = True,
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
        *,
        identifiers_one_based: bool = False,
        age_label_mode: str = "bucket",   # "code" | "bucket" | "year"
        remap_occ_1_based: bool = False,
        debug_sample: bool = False,
    ):
        self.base_dataset = base_dataset
        self.target_attr = target_attr
        self.include_identifiers = include_identifiers
        self.include_rating = include_rating
        self.identifiers_one_based = identifiers_one_based
        self.age_label_mode = age_label_mode
        self.remap_occ_1_based = remap_occ_1_based
        self.debug_sample = debug_sample

        # --- detect num_users/items ---
        root_ds = getattr(base_dataset, "dataset", base_dataset)
        detected_users, detected_items = None, None

        gvs = getattr(root_ds, "get_vocab_sizes", None)
        if callable(gvs):
            try:
                vs = gvs()
                detected_users = int(vs.get("num_users")) if vs and "num_users" in vs else None
                detected_items = int(vs.get("num_items")) if vs and "num_items" in vs else None
            except Exception:
                pass

        if detected_users is None:
            try:
                detected_users = int(len(getattr(root_ds, "user2idx")))
            except Exception:
                pass
        if detected_items is None:
            try:
                detected_items = int(len(getattr(root_ds, "movie2idx")))
            except Exception:
                pass

        # safe defaults
        if detected_users is None:
            detected_users = 6040
        if detected_items is None:
            detected_items = 3952

        self.num_users = int(num_users) if num_users is not None else detected_users
        self.num_items = int(num_items) if num_items is not None else detected_items
        self._u_den = max(1, self.num_users - 1)
        self._i_den = max(1, self.num_items - 1)

        # ---- support genre as a target too ----
        self._supported_attrs = {"gender", "occupation", "age", "genre"}
        if self.target_attr not in self._supported_attrs:
            raise ValueError(f"Unsupported target_attr: {self.target_attr}")

        # --- detect genre dimension from the first sample, else default 19 ---
        self._genre_dim = 19
        try:
            first = base_dataset[0]
            if isinstance(first, dict) and "genre" in first:
                g = first["genre"]
                self._genre_dim = int(g.numel()) if torch.is_tensor(g) else int(len(g))
        except Exception:
            pass

        self._logged_once = False
        self._warned_occ = False
        self._age_codes = torch.tensor([1., 18., 25., 35., 45., 50., 56.], dtype=torch.float32)

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def _get_multi(item, names, default_tensor):
        if isinstance(item, dict):
            for n in names:
                if n in item:
                    return item[n]
        return default_tensor

    def _get_safe(self, item, name, default_tensor):
        if isinstance(item, dict) and name in item:
            return item[name]
        return default_tensor

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        feats = []

        dev = None
        # try to pick a device early (from any present tensor)
        for k in ("user_id", "item_id", "age", "gender", "occupation", "genre", "rating", "pred_rating"):
            if isinstance(item, dict) and k in item and torch.is_tensor(item[k]):
                dev = item[k].device
                break
        if dev is None:
            dev = torch.device("cpu")

        # identifiers (optional)
        if self.include_identifiers:
            user_raw = self._get_safe(item, "user_id", torch.tensor(0.0, device=dev)).float()
            item_raw = self._get_safe(item, "item_id", torch.tensor(0.0, device=dev)).float()
            if self.identifiers_one_based:
                user_raw = user_raw - 1.0
                item_raw = item_raw - 1.0
            user_id = (user_raw.unsqueeze(0) / float(self._u_den)).clamp_(0, 1)
            item_id = (item_raw.unsqueeze(0) / float(self._i_den)).clamp_(0, 1)
            feats.extend([user_id, item_id])

        # rating (optional) handle 'pred_rating' or 'rating'
        if self.include_rating:
            r_raw = self._get_multi(item, ["pred_rating", "rating"], torch.tensor(0.0, device=dev))
            r_raw = r_raw.float()
            # If value looks already normalized [0,1], keep; else divide by 5
            if torch.isfinite(r_raw).all():
                r01 = r_raw if (r_raw.min() >= 0.0 and r_raw.max() <= 1.0) else (r_raw / 5.0)
            else:
                r01 = torch.zeros_like(r_raw)
            feats.append(r01.unsqueeze(0).clamp_(0, 1))

        # non-target attributes (scaled like WGAN inputs / synthetic canonical)
        if self.target_attr != "age":
            age_raw = self._get_safe(item, "age", torch.tensor(0.0, device=dev))
            feats.append((age_raw.float().unsqueeze(0) / 100.0).clamp_(0, 1))
        if self.target_attr != "occupation":
            occ_raw = self._get_safe(item, "occupation", torch.tensor(0.0, device=dev))
            feats.append((occ_raw.float().unsqueeze(0) / 20.0).clamp_(0, 1))
        if self.target_attr != "gender":
            gen_raw = self._get_safe(item, "gender", torch.tensor(0, device=dev))
            feats.append(gen_raw.float().unsqueeze(0))

        # genre vector (multi-hot) — exclude from X if it's the target
        g_default = torch.zeros(self._genre_dim, dtype=torch.float32, device=dev)
        genre_vec = self._get_safe(item, "genre", g_default).float()
        if genre_vec.numel() != self._genre_dim:
            if genre_vec.numel() < self._genre_dim:
                pad = torch.zeros(self._genre_dim - genre_vec.numel(), device=dev)
                genre_vec = torch.cat([genre_vec, pad], dim=0)
            else:
                genre_vec = genre_vec[: self._genre_dim]
        if self.target_attr != "genre":
            feats.append(genre_vec)

        feature = torch.cat(feats, dim=0).to(torch.float32)

        # expected dim (don’t count genre if predicting genre)
        id_dims = 2 if self.include_identifiers else 0
        rating_dims = 1 if self.include_rating else 0
        age_dims = 0 if self.target_attr == "age" else 1
        occ_dims = 0 if self.target_attr == "occupation" else 1
        gen_dims = 0 if self.target_attr == "gender" else 1
        genre_dims = 0 if self.target_attr == "genre" else self._genre_dim
        expected_dim = id_dims + rating_dims + age_dims + occ_dims + gen_dims + genre_dims

        if feature.numel() != expected_dim:
            raise RuntimeError(
                f"Feature dim mismatch: got {feature.numel()}, expected {expected_dim} "
                f"(ids={id_dims}, rating={rating_dims}, age={age_dims}, occ={occ_dims}, "
                f"gender={gen_dims}, genre={genre_dims})."
            )

        if self.debug_sample and not self._logged_once:
            self._logged_once = True
            print(f"[AttributeInferenceDataset] feature_dim={feature.numel()} | expected_dim={expected_dim} | "
                  f"include_identifiers={self.include_identifiers} | include_rating={self.include_rating} | "
                  f"target={self.target_attr} | genre_dim={self._genre_dim}")

        # ---- labels ----
        if self.target_attr == "occupation":
            occ_lab = self._get_safe(item, "occupation", torch.tensor(0, device=dev)).long()
            if self.remap_occ_1_based and torch.any((occ_lab >= 1) & (occ_lab <= 21)):
                if not self._warned_occ:
                    print("[AttributeInferenceDataset] Remapping occupation labels 1..21 -> 0..20")
                    self._warned_occ = True
                occ_lab = (occ_lab - 1).clamp_(0, 20)
            label = occ_lab.to(torch.int64).clamp_(0, 20)

        elif self.target_attr == "gender":
            gen_lab = self._get_safe(item, "gender", torch.tensor(0, device=dev)).long()
            label = gen_lab.to(torch.int64).clamp_(0, 1)

        elif self.target_attr == "genre":
            # multilabel (multi-hot) target
            g_default = torch.zeros(self._genre_dim, dtype=torch.float32, device=dev)
            genre_lab = self._get_safe(item, "genre", g_default).float()
            if genre_lab.numel() != self._genre_dim:
                if genre_lab.numel() < self._genre_dim:
                    pad = torch.zeros(self._genre_dim - genre_lab.numel(), device=dev)
                    genre_lab = torch.cat([genre_lab, pad], dim=0)
                else:
                    genre_lab = genre_lab[: self._genre_dim]
            label = genre_lab

        else:  # age
            age_val = self._get_safe(item, "age", torch.tensor(0.0, device=dev)).float()
            if self.age_label_mode == "bucket":
                codes = self._age_codes.to(age_val.device)
                idx = torch.argmin(torch.abs(age_val.view(-1, 1) - codes.view(1, -1)), dim=1)
                label = idx.long().view_as(age_val).long()
            elif self.age_label_mode == "year":
                label = age_val  # raw years regression
            else:  # "code"
                label = age_val  # ML-1M code regression

        return feature, label
    
class SurrogatePairsFromCanonical25(torch.utils.data.Dataset):
    """
    Wraps a dataset that returns a 25-D canonical vector in [0,1]:
      [user_id, item_id, age, gender, occupation_scalar, genre(19), rating/5]
    Emits (x, y) where:
      x = first 24 dims (no rating), shape [24]
      y = rating/5 in [0,1], shape [1]
    """
    def __init__(self, base_01_dataset):
        self.base = base_01_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        v = self.base[idx]
        if v.ndim != 1 or v.shape[0] not in (25, 45):
            raise RuntimeError(f"Expected 25-D canonical vector, got shape {tuple(v.shape)}")
        if v.shape[0] != 25:
            raise RuntimeError("SurrogatePairsFromCanonical25 expects scalar occupation (25-D). Set use_onehot_occ=False.")
        x = v[:24].to(torch.float32)      # [24] in [0,1]
        y = v[24:25].to(torch.float32)    # [1]  in [0,1]
        return x, y
