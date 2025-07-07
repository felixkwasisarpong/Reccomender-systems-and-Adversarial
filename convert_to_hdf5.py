import pandas as pd
import numpy as np
import h5py
import os
from collections import defaultdict

# ========== CONFIG ==========
base_dir = "ml-100k"
output_file = "movielens_100k_with_attrs.hdf5"
normalize_age = True
use_onehot_occupation = False  # Set to True to use one-hot encoding
drop_gender = False
drop_age = False
drop_occupation = False

# ========== 1. Load Files ==========
ratings = pd.read_csv(os.path.join(base_dir, "u.data"), sep="\t", names=["userId", "movieId", "rating", "timestamp"])
users = pd.read_csv(os.path.join(base_dir, "u.user"), sep="|", names=["userId", "age", "gender", "occupation", "zip"])
movies = pd.read_csv(os.path.join(base_dir, "u.item"), sep="|", encoding='latin-1',
                     names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL"] + [f"genre_{i}" for i in range(19)],
                     usecols=list(range(24)))

# ========== 2. Merge ==========
df = ratings.merge(users, on="userId").merge(movies, on="movieId")

# ========== 3. Genre One-Hot ==========
genre_matrix = df[[f"genre_{i}" for i in range(19)]].values.astype(np.uint8)

# ========== 4. Gender ==========
if not drop_gender:
    df["gender_encoded"] = (df["gender"] == "M").astype(np.uint8)

# ========== 5. Age ==========
if not drop_age:
    if normalize_age:
        df["age_normalized"] = (df["age"] - df["age"].mean()) / df["age"].std()
    else:
        df["age_normalized"] = df["age"].astype(np.float32)

# ========== 6. Occupation ==========
if not drop_occupation:
    if use_onehot_occupation:
        occupation_dummies = pd.get_dummies(df["occupation"], prefix="occ")
    else:
        occ_map = {occ: idx for idx, occ in enumerate(sorted(df["occupation"].unique()))}
        df["occupation_encoded"] = df["occupation"].map(occ_map).astype(np.uint8)

# ========== 7. Final Arrays ==========
user_ids = df["userId"].astype(np.int32).values
movie_ids = df["movieId"].astype(np.int32).values
ratings_arr = df["rating"].astype(np.float32).values

# ========== 8. Write to HDF5 ==========
with h5py.File(output_file, "w") as f:
    f.create_dataset("user_ids", data=user_ids)
    f.create_dataset("item_ids", data=movie_ids)
    f.create_dataset("ratings", data=ratings_arr)
    f.create_dataset("genre_onehot", data=genre_matrix)

    if not drop_gender:
        f.create_dataset("gender", data=df["gender_encoded"].values.astype(np.uint8))
    if not drop_age:
        f.create_dataset("age", data=df["age_normalized"].values.astype(np.float32))
    if not drop_occupation:
        if use_onehot_occupation:
            f.create_dataset("occupation", data=occupation_dummies.values.astype(np.uint8))
        else:
            f.create_dataset("occupation", data=df["occupation_encoded"].values.astype(np.uint8))

    f.attrs["global_mean"] = float(np.mean(ratings_arr))
    f.attrs["num_genres"] = 19

print(f"✅ Saved: {output_file}")