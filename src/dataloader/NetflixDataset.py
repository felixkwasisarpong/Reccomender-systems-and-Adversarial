import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class NetflixDataset(Dataset):
    def __init__(self, data_dir, mode='train', max_samples=None, test_users_ratio=0.2, random_state=42):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.max_samples = max_samples
        self.test_users_ratio = test_users_ratio
        self.random_state = random_state

        h5_files = [f for f in os.listdir(data_dir) if f.endswith(('.hdf5', '.h5'))]
        if not h5_files:
            raise FileNotFoundError(f"No .hdf5 or .h5 file found in {data_dir}")
        self.h5_path = os.path.join(data_dir, h5_files[0])

        with h5py.File(self.h5_path, 'r') as file:
            self.global_mean = float(file.attrs['global_mean']) if 'global_mean' in file.attrs else 0.0
            self.num_ratings = int(file.attrs['num_ratings']) if 'num_ratings' in file.attrs else 0

            ratings = file['ratings']
            self.user_ids = ratings['user_id'][:].astype(np.int64)
            self.movie_ids = ratings['movie_id'][:].astype(np.int32)
            self.ratings = ratings['rating'][:].astype(np.float32)

            self.users = np.unique(self.user_ids)
            self.movies = np.unique(self.movie_ids)

            self.user2idx = {uid: idx for idx, uid in enumerate(self.users)}
            self.movie2idx = {mid: idx for idx, mid in enumerate(self.movies)}

            rng = np.random.default_rng(random_state)
            user_list = np.array(self.users)
            rng.shuffle(user_list)
            test_size = int(len(user_list) * test_users_ratio)

            if mode == 'train':
                selected_users = set(user_list[test_size:])
            elif mode in ['val', 'test']:
                selected_users = set(user_list[:test_size])
            else:
                raise ValueError(f"Invalid mode: {mode}")

            self.indices = np.array([
                i for i, uid in enumerate(self.user_ids)
                if uid in selected_users
            ], dtype=np.int64)

            if max_samples:
                self.indices = self.indices[:max_samples]

            # Show top 4 data entries
            # print(f"\n[Dataset Stats: Mode={self.mode}]")
            # for i in self.indices[4:]:
            #     print(f"UserID: {self.user_ids[i]}, MovieID: {self.movie_ids[i]}, Rating: {self.ratings[i]}")

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as file:
            ratings = file['ratings']
            i = self.indices[idx]
            row = ratings[i]
            user_idx = self.user2idx[int(row['user_id'])]
            movie_idx = self.movie2idx[int(row['movie_id'])]
            rating = float(row['rating'])
            return (
                torch.tensor(user_idx, dtype=torch.int64),
                torch.tensor(movie_idx, dtype=torch.int32),
                torch.tensor(rating, dtype=torch.float32)
            )

    def __len__(self):
        return len(self.indices)

    def get_num_users(self):
        return len(self.user2idx)

    def get_num_movies(self):
        return len(self.movie2idx)

    def get_global_mean(self):
        return self.global_mean

