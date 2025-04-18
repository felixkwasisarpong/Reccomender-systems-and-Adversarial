#!/usr/bin/env python3
import os
import h5py
import numpy as np
from tqdm import tqdm

def convert_netflix_to_hdf5(data_dir, hdf5_file):
    # Define the structured data type
    dt = np.dtype([
        ('user_id', 'i4'),
        ('movie_id', 'i4'),
        ('rating', 'f4'),
        ('date', 'S10')  # Fixed-length string YYYY-MM-DD
    ])
    
    user_id_to_idx = {}
    next_user_idx = 0
    total_ratings = 0
    rating_sum = 0.0
    batch_size = 100000
    batch = np.zeros(batch_size, dtype=dt)
    batch_idx = 0

    with h5py.File(hdf5_file, 'w') as hf:
        dset = hf.create_dataset(
            'ratings',
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            compression='gzip',
            chunks=True
        )

        for filename in tqdm(sorted(os.listdir(data_dir)), desc="Processing files"):
            if not filename.endswith('.txt'):
                continue

            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue

                # First line is movie ID
                try:
                    movie_id = int(lines[0].strip().replace(':', ''))
                except ValueError:
                    continue  # Skip if movie ID is not valid

                for line in lines[1:]:
                    try:
                        user_id, rating, date = line.strip().split(',')
                        user_id = int(user_id)
                        rating = float(rating)
                    except ValueError:
                        continue  # Skip malformed lines

                    if not (1.0 <= rating <= 5.0):
                        continue

                    # Map user ID to index
                    if user_id not in user_id_to_idx:
                        user_id_to_idx[user_id] = next_user_idx
                        next_user_idx += 1

                    batch[batch_idx] = (
                        user_id_to_idx[user_id],
                        movie_id,
                        rating,
                        np.string_(date)
                    )
                    batch_idx += 1
                    total_ratings += 1
                    rating_sum += rating

                    # Flush batch
                    if batch_idx == batch_size:
                        dset.resize((dset.shape[0] + batch_size,))
                        dset[-batch_size:] = batch
                        batch_idx = 0

        # Final flush
        if batch_idx > 0:
            dset.resize((dset.shape[0] + batch_idx,))
            dset[-batch_idx:] = batch[:batch_idx]

        hf.attrs['global_mean'] = (rating_sum / total_ratings) if total_ratings > 0 else 0.0
        hf.attrs['num_ratings'] = total_ratings
        hf.attrs['num_users'] = len(user_id_to_idx)
        hf.attrs['num_movies'] = expected_movie_id = len([f for f in os.listdir(data_dir) if f.endswith('.txt')])

        print(f"\n✅ Conversion complete:")
        print(f"- Unique users: {len(user_id_to_idx):,}")
        print(f"- Movies processed: {hf.attrs['num_movies']:,}")
        print(f"- Total ratings: {total_ratings:,}")
        print(f"- Global average rating: {hf.attrs['global_mean']:.2f}")
        print(f"📁 Saved to {hdf5_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert Netflix dataset to HDF5 format')
    parser.add_argument("--folder_path", required=True, help="Path to the training_set folder")
    parser.add_argument("--output", default="netflix_data.hdf5", help="Output HDF5 filename")
    args = parser.parse_args()

    if not args.output.endswith(".hdf5"):
        args.output += ".hdf5"

    convert_netflix_to_hdf5(args.folder_path, args.output)
