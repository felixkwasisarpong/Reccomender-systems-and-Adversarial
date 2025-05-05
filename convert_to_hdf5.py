import h5py
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

def parse_netflix_file(file_path, has_ratings=True):
    """Parse Netflix file (handles both training and test formats)"""
    data = []
    current_user = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc=f"Processing {os.path.basename(file_path)}"):
        line = line.strip()
        if not line:
            continue
            
        if line.endswith(':'):
            current_user = int(line[:-1])
        else:
            parts = line.split(',')
            if has_ratings and len(parts) == 3:  # Training format
                movie_id = int(parts[0])
                rating = float(parts[1])
                date = datetime.strptime(parts[2], "%Y-%m-%d")
                data.append((current_user, movie_id, rating, date))
            elif not has_ratings and len(parts) == 2:  # Test/probe format
                movie_id = int(parts[0])
                date = datetime.strptime(parts[1], "%Y-%m-%d")
                data.append((current_user, movie_id, -1.0, date))  # -1 as placeholder
    
    return data

def create_hdf5_file(data, output_path, dataset_type='train'):
    """Create HDF5 file with metadata"""
    user_ids = np.array([x[0] for x in data], dtype=np.int64)
    movie_ids = np.array([x[1] for x in data], dtype=np.int32)
    ratings = np.array([x[2] for x in data], dtype=np.float32)
    dates = np.array([x[3].timestamp() for x in data], dtype=np.float64)
    
    with h5py.File(output_path, 'w') as f:
        # Store dataset type as attribute
        f.attrs['dataset_type'] = dataset_type
        
        # Only calculate global mean for training data
        if dataset_type == 'train':
            f.attrs['global_mean'] = np.mean(ratings)
            f.attrs['num_ratings'] = len(ratings)
        else:
            f.attrs['global_mean'] = 0.0
            f.attrs['num_ratings'] = len(ratings)
        
        # Create datasets
        ratings_group = f.create_group('ratings')
        ratings_group.create_dataset('user_id', data=user_ids)
        ratings_group.create_dataset('movie_id', data=movie_ids)
        ratings_group.create_dataset('rating', data=ratings)
        ratings_group.create_dataset('date', data=dates)

def process_all_files( probe_path, qualifying_path, output_dir):
    """Process all three Netflix files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Process training set
    # print("\nProcessing training set...")
    # train_data = parse_netflix_file(train_path, has_ratings=True)
    # train_output = os.path.join(output_dir, "train_data.hdf5")
    # create_hdf5_file(train_data, train_output, dataset_type='train')
    
    # 2. Process probe set (validation)
    print("\nProcessing probe set (validation)...")
    probe_data = parse_netflix_file(probe_path, has_ratings=False)
    probe_output = os.path.join(output_dir, "val_data.hdf5")
    create_hdf5_file(probe_data, probe_output, dataset_type='val')
    
    # 3. Process qualifying set (test)
    print("\nProcessing qualifying set (test)...")
    qualifying_data = parse_netflix_file(qualifying_path, has_ratings=False)
    qualifying_output = os.path.join(output_dir, "test_data.hdf5")
    create_hdf5_file(qualifying_data, qualifying_output, dataset_type='test')
    
    print(f"\nConversion complete. Files saved to {output_dir}:")
    # print(f"- Training: {train_output}")
    print(f"- Validation: {probe_output}")
    print(f"- Test: {qualifying_output}")

if __name__ == "__main__":
    # Configure paths
    data_dir = "netflix_data"  # Directory containing raw files
    output_dir = "netflix_data"  # Where to save HDF5 files
    
    # train_file = os.path.join(data_dir, "training_set.txt")  # Original training data
    probe_file = os.path.join(data_dir, "probe.txt")        # Validation set
    qualifying_file = os.path.join(data_dir, "qualifying.txt")  # Final test set
    
    # Run conversion
    process_all_files(probe_file, qualifying_file, output_dir)