import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torch.optim.lr_scheduler import ReduceLROnPlateau
 
class NetflixDataset(Dataset):
    def __init__(
        self,
        h5_path,
        mode='train',
        max_samples=None,
        split_ratio=0.9,
        random_state=42,
        return_attrs: bool = True,
        plot_eda: bool = False,
        eda_out_dir: str = "./eda"
    ):
        super().__init__()
        self.h5_path = h5_path
        self.mode = mode
        self.max_samples = max_samples
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.return_attrs = return_attrs
        self.plot_eda = plot_eda
        self.eda_out_dir = eda_out_dir

        # with h5py.File(self.h5_path, 'r') as file:
        #     self.global_mean = float(file.attrs.get('global_mean', 0.0))
        #     self.user_ids_all = file['user_ids'][:].astype(np.int64)
        #     self.movie_ids_all = file['item_ids'][:].astype(np.int32)
        #     self.ratings_all = file['ratings'][:].astype(np.float32)

        #     self.has_attrs = all(key in file for key in ['gender', 'age', 'occupation', 'genre_onehot'])
        #     if self.has_attrs:
        #         self.gender_all = file['gender'][:]
        #         self.age_all = file['age'][:]
        #         self.occupation_all = file['occupation'][:]
        #         self.genre_all = file['genre_onehot'][:]
        with h5py.File(self.h5_path, 'r') as file:
  
            if 'predictions' in file:
             
                # Structured array format (like cus_weak.hdf5)
                preds = file['predictions']
                self.user_ids_all = preds['user_id'][:].astype(np.int64)
                self.movie_ids_all = preds['movie_id'][:].astype(np.int32)
                self.ratings_all = preds['pred_rating'][:].astype(np.float32)
                self.gender_all = preds['gender'][:].astype(np.int64)
                self.age_all = preds['age'][:].astype(np.float32)
                self.occupation_all = preds['occupation'][:].astype(np.int64)
                self.genre_all = preds['genre'][:].astype(np.float32)
                self.has_attrs = True
            else:
                # Flat layout (like movielens_100k_with_attrs.hdf5)
                self.user_ids_all = file['user_ids'][:].astype(np.int64)
                self.movie_ids_all = file['item_ids'][:].astype(np.int32)
                self.ratings_all = file['ratings'][:].astype(np.float32)
                if all(k in file for k in ['gender', 'age', 'occupation', 'genre_onehot']):
                    self.gender_all = file['gender'][:].astype(np.int64)
                    self.age_all = file['age'][:].astype(np.float32)
                    self.occupation_all = file['occupation'][:].astype(np.int64)
                    self.genre_all = file['genre_onehot'][:].astype(np.float32)
                    self.has_attrs = True
                else:
                    self.has_attrs = False


        indices = np.arange(len(self.user_ids_all))
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)

        if self.mode == 'train':
            split_idx = int(len(indices) * self.split_ratio)
            indices = indices[:split_idx]
        elif self.mode == 'val':
            split_idx = int(len(indices) * self.split_ratio)
            indices = indices[split_idx:]

        if max_samples is not None and max_samples > 0:
            indices = indices[:max_samples]

        self.user_ids = self.user_ids_all[indices]
        self.movie_ids = self.movie_ids_all[indices]
        self.ratings = self.ratings_all[indices]

        if self.has_attrs:
            self.genders = self.gender_all[indices]
            self.ages = self.age_all[indices]
            self.occupations = self.occupation_all[indices]
            self.genres = self.genre_all[indices]

        self.user2idx = {uid: idx for idx, uid in enumerate(np.unique(self.user_ids_all))}
        self.movie2idx = {mid: idx for idx, mid in enumerate(np.unique(self.movie_ids_all))}
        self.user_ids = np.array([self.user2idx[uid] for uid in self.user_ids], dtype=np.int64)
        self.movie_ids = np.array([self.movie2idx[mid] for mid in self.movie_ids], dtype=np.int64)

    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'item_id': torch.tensor(self.movie_ids[idx], dtype=torch.long),
            'gender': torch.tensor(self.genders[idx], dtype=torch.long),
            'age': torch.tensor(self.ages[idx], dtype=torch.float32),
            'occupation': torch.tensor(self.occupations[idx], dtype=torch.long),
            'genre': torch.tensor(self.genres[idx], dtype=torch.float32),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.user_ids)

    def get_vocab_sizes(self):
        return {
            'num_users': len(self.user2idx),
            'num_items': len(self.movie2idx),
            'num_genders': int(self.genders.max()) + 1,
            'num_occupations': int(self.occupations.max()) + 1,
            'genre_dim': self.genres.shape[1],
        }

    @staticmethod
    def plot_combined_eda(user_ids, movie_ids, ratings, out_path="training.png"):
        sns.set(style="whitegrid")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        sns.histplot(ratings, bins=5, kde=False, ax=axs[0, 0])
        axs[0, 0].set_title("Ratings Distribution")
        axs[0, 0].set_xlabel("Rating")
        axs[0, 0].set_ylabel("Count")

        user_counts = np.bincount(user_ids)
        axs[0, 1].hist(user_counts[user_counts > 0], bins=50, log=True)
        axs[0, 1].set_title("User Activity (log)")
        axs[0, 1].set_xlabel("# Ratings per User")

        movie_counts = np.bincount(movie_ids)
        axs[1, 0].hist(movie_counts[movie_counts > 0], bins=50, log=True)
        axs[1, 0].set_title("Movie Popularity (log)")
        axs[1, 0].set_xlabel("# Ratings per Movie")

        axs[1, 1].hist(np.unique(user_ids), bins=100)
        axs[1, 1].set_title("Unique User ID Distribution")
        axs[1, 1].set_xlabel("User ID Index")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"\U0001f4ca Saved combined EDA plot to {out_path}")
