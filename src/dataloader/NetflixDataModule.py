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


import os
from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .NetflixDataset import NetflixDataset

try:
    from opacus.utils.uniform_sampler import UniformWithReplacementSampler as DPDataLoader
except ImportError:
    DPDataLoader = None

class NetflixDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 1024, num_workers: int = 4, max_samples: Optional[int] = None,pred_file: Optional[str] = "dp_mid",        train_split_ratio: float = 0.8,
        val_split_ratio: float = 0.1, random_state: int = 42, return_attrs: bool = True, enable_dp: bool = False):
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

    def setup(self, stage: Optional[str] = None):
        full_dataset = NetflixDataset(
            h5_path=os.path.join(self.data_dir, f"{self.pred_file}.hdf5"),
            max_samples=self.max_samples,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode='full'  # Load everything once
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

        # Predict dataset is completely separate
        self.predict_dataset = NetflixDataset(
            h5_path=os.path.join(self.data_dir, f"{self.pred_file}.hdf5"),
            max_samples=None,
            random_state=self.random_state,
            return_attrs=self.return_attrs,
            mode='predict'
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
        if self.enable_dp:
            assert DPDataLoader is not None, "Opacus must be installed for DP training."
            return DPDataLoader(
                dataset=self.train_dataset,
                sample_rate=self.batch_size / len(self.train_dataset),
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                generator=torch.Generator().manual_seed(self.random_state)
            )
        else:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True
            )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def class_attack_train_dataloader(self):
        return DataLoader(self.class_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,persistent_workers=True)

    def class_attack_val_dataloader(self):
        return DataLoader(self.class_val, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers,persistent_workers=True)

    def class_attack_test_dataloader(self):
        return DataLoader(self.class_test, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers,persistent_workers=True)


    def attack_dataloader(self):
        return DataLoader(
            self.attack_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )


    def mia_dataloaders(self, member_path, nonmember_path):
        member_ds = NetflixDataset(member_path, mode='train', return_attrs=self.return_attrs)
        nonmember_ds = NetflixDataset(nonmember_path, mode='test', return_attrs=self.return_attrs)

        mia_ds = MIADataset(member_ds, nonmember_ds)
        return [
            DataLoader(
                mia_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        ]

    def get_num_users(self):
        return self.train_dataset.dataset.get_num_users()

    def get_num_movies(self):
        return self.train_dataset.dataset.get_num_movies()


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

        # Always include these
        example = {
            'user_id': data['user_id'],
            'item_id': data['item_id'],
            'rating': data['rating'],
            'label': torch.tensor(label, dtype=torch.float32)
        }

        # Always include attrs if return_attrs is True
        if 'gender' in data:
            example['gender'] = data['gender']
            example['age'] = data['age']
            example['occupation'] = data['occupation']
            example['genre'] = data['genre']

        return example
    
class WGANInputDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        user_id    = item['user_id'].float().unsqueeze(0) / 6000       # ✅ if max user_id < 6000
        item_id    = item['item_id'].float().unsqueeze(0) / 4000       # ✅ if max item_id < 4000
        gender     = item['gender'].float().unsqueeze(0)               # ✅ already binary
        age        = item['age'].unsqueeze(0) / 100                    # ✅ assume age < 100
        occupation = item['occupation'].float().unsqueeze(0) / 20      # ✅ assume occ < 20
        genre      = item['genre']                                     # ✅ already in [0, 1]
        rating     = item['rating'].unsqueeze(0) / 5.0                 # ✅ if original ratings in [1, 5]

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
        genre      = item['genre']  # [19-dim]

        # Compose input feature vector
        feature = torch.cat([user_id, item_id, rating, age, occupation, genre], dim=0)

        # Target label
        if self.target_attr == "age":
            label = item["age"].float()
        elif self.target_attr == "gender":
            label = item["gender"].long()
        elif self.target_attr == "occupation":
            label = item["occupation"].long()
        else:
            raise ValueError(f"Unsupported target_attr: {self.target_attr}")

        return feature, label
