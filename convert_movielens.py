import pandas as pd
import numpy as np
import h5py
import os

# Step 1: Load MovieLens 100k
ml_path = "ml-100k/u.data"  # Update this if needed
df = pd.read_csv(ml_path, sep='\t', names=["user_id", "movie_id", "rating", "timestamp"])

# Step 2: Cast to appropriate types
df['user_id'] = df['user_id'].astype(np.int64)
df['movie_id'] = df['movie_id'].astype(np.int32)
df['rating'] = df['rating'].astype(np.float32)

# Step 3: Calculate metadata
global_mean = df['rating'].mean()
num_ratings = len(df)

# Step 4: Write to HDF5
output_path = "movielens_100k_formatted.hdf5"
with h5py.File(output_path, 'w') as f:
    ratings_group = f.create_group('ratings')
    ratings_group.create_dataset('user_id', data=df['user_id'].values)
    ratings_group.create_dataset('movie_id', data=df['movie_id'].values)
    ratings_group.create_dataset('rating', data=df['rating'].values)

    f.attrs['global_mean'] = global_mean
    f.attrs['num_ratings'] = num_ratings

print(f"HDF5 file saved at: {output_path}")
