import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class NetflixDataset(Dataset):
    def __init__(self, h5_path, mode='train', max_samples=None, split_ratio=0.9, random_state=42):
        """
        Args:
            h5_path: Path to HDF5 file
            mode: 'train', 'val', or 'test'
            max_samples: Optional limit on samples
            split_ratio: Ratio of training data (only used for train/val modes)
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.h5_path = h5_path
        self.mode = mode
        self.max_samples = max_samples
        self.split_ratio = split_ratio
        self.random_state = random_state

        # Load data from HDF5
        with h5py.File(self.h5_path, 'r') as file:
            self.global_mean = float(file.attrs.get('global_mean', 0.0))
            self.num_ratings = int(file.attrs.get('num_ratings', 0))
            
            ratings = file['ratings']
            user_ids = ratings['user_id'][:].astype(np.int64)
            movie_ids = ratings['movie_id'][:].astype(np.int32)
            ratings = ratings['rating'][:].astype(np.float32)

        # Create mappings
        self.users = np.unique(user_ids)
        self.movies = np.unique(movie_ids)
        self.user2idx = {uid: idx for idx, uid in enumerate(self.users)}
        self.movie2idx = {mid: idx for idx, mid in enumerate(self.movies)}

        # Handle splits differently for test vs train/val
        if mode == 'test':
            # Use all test data as-is
            self.indices = np.arange(len(user_ids))
        else:
            # Split train/val data
            rng = np.random.default_rng(self.random_state)
            indices = np.arange(len(user_ids))
            rng.shuffle(indices)
            split_idx = int(len(indices) * self.split_ratio)
            self.indices = indices[:split_idx] if mode == 'train' else indices[split_idx:]

        # Apply indices
        self.user_ids = user_ids[self.indices]
        self.movie_ids = movie_ids[self.indices]
        self.ratings = ratings[self.indices]

        # Subsample if requested
        if self.max_samples is not None and self.max_samples > 0:
            rng = np.random.default_rng(self.random_state)
            sample_size = min(self.max_samples, len(self.user_ids))
            sample_indices = rng.choice(len(self.user_ids), size=sample_size, replace=False)
            self.user_ids = self.user_ids[sample_indices]
            self.movie_ids = self.movie_ids[sample_indices]
            self.ratings = self.ratings[sample_indices]

        self._log_stats()

    def __getitem__(self, idx):
        return (
            torch.tensor(self.user2idx[self.user_ids[idx]], dtype=torch.int64),
            torch.tensor(self.movie2idx[self.movie_ids[idx]], dtype=torch.int64),
            torch.tensor(self.ratings[idx], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.user_ids)

    def _log_stats(self):
        print(f"\n[Dataset {self.mode}]")
        print(f"  File: {os.path.basename(self.h5_path)}")
        print(f"  Samples: {len(self.user_ids):,}")
        print(f"  Unique users: {len(self.users):,}")
        print(f"  Unique movies: {len(self.movies):,}")
        print(f"  Global mean: {self.global_mean:.2f}")

    def get_num_users(self):
        return len(self.users)

    def get_num_movies(self):
        return len(self.movies)