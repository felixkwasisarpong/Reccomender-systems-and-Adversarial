import os
import numpy as np
import pandas as pd
import h5py

# === 1. Define standard genre mapping (based on 100k) ===
standard_genres = [
    'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'IMAX'
]
genre_to_idx = {g: i for i, g in enumerate(standard_genres)}

def encode_genres(genre_str):
    vec = np.zeros(len(genre_to_idx), dtype=np.uint8)
    for g in genre_str.split('|'):
        if g in genre_to_idx:
            vec[genre_to_idx[g]] = 1
    return vec

# === 2. Load MovieLens 1M files ===
base_dir = "ml-1m"  # path to folder with users.dat, ratings.dat, movies.dat

users = pd.read_csv(
    os.path.join(base_dir, 'users.dat'),
    sep='::', engine='python',
    names=['userId', 'gender', 'age', 'occupation', 'zip'],
    encoding='latin-1'
)

ratings = pd.read_csv(
    os.path.join(base_dir, 'ratings.dat'),
    sep='::', engine='python',
    names=['userId', 'movieId', 'rating', 'timestamp'],
    encoding='latin-1'
)

movies = pd.read_csv(
    os.path.join(base_dir, 'movies.dat'),
    sep='::', engine='python',
    names=['movieId', 'title', 'genres'],
    encoding='latin-1'
)

# === 3. Merge datasets ===
df = ratings.merge(users, on='userId').merge(movies, on='movieId')

# === 3.5 Remap userId and movieId to zero-based contiguous indices compatible with ML-100K limits ===
# This remapping ensures compatibility with models trained on ML-100K embeddings.
num_users_100k = 943
num_items_100k = 1682

# Map original userIds to zero-based indices
unique_user_ids = df['userId'].unique()
user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_user_ids)}
df['userId'] = df['userId'].map(user_id_map)

# For userIds exceeding ML-100K limit, remap to random valid ID within range
df.loc[df['userId'] >= num_users_100k, 'userId'] = np.random.randint(0, num_users_100k, size=(df['userId'] >= num_users_100k).sum())

# Map original movieIds to zero-based indices
unique_movie_ids = df['movieId'].unique()
movie_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids)}
df['movieId'] = df['movieId'].map(movie_id_map)

# For movieIds exceeding ML-100K limit, remap to random valid ID within range
df.loc[df['movieId'] >= num_items_100k, 'movieId'] = np.random.randint(0, num_items_100k, size=(df['movieId'] >= num_items_100k).sum())

# === 4. Encode features ===
user_ids = df['userId'].astype(np.int32).values
item_ids = df['movieId'].astype(np.int32).values
ratings_arr = df['rating'].astype(np.float32).values
genders = (df['gender'] == 'M').astype(np.uint8).values    # M → 1, F → 0
ages = df['age'].astype(np.int32).values                   # categorical age code
occupations = df['occupation'].astype(np.int32).values     # occupation ID
genre_matrix = np.stack(df['genres'].apply(encode_genres)) # [N, 19]

# === 5. Compute global mean ===
global_mean = np.mean(ratings_arr)

# === 6. Write to HDF5 ===
output_path = "movielens_1m_with_attrs.hdf5"
with h5py.File(output_path, "w") as f:
    f.create_dataset("user_ids", data=user_ids)
    f.create_dataset("item_ids", data=item_ids)
    f.create_dataset("ratings", data=ratings_arr)
    f.create_dataset("gender", data=genders)
    f.create_dataset("age", data=ages)
    f.create_dataset("occupation", data=occupations)
    f.create_dataset("genre_onehot", data=genre_matrix)

    # Add metadata
    f.attrs["global_mean"] = float(global_mean)
    f.attrs["num_genres"] = len(standard_genres)
    f.attrs["genre_labels"] = np.array(standard_genres, dtype="S")

print(f"✅ Saved: {output_path}")
print(f"🎬 Genre matrix shape: {genre_matrix.shape}")
print(f"🌍 Global mean rating: {global_mean:.4f}")