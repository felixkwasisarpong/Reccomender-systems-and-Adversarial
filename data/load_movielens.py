import numpy as np
import torch
from surprise import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def load_movielens():
    # Load MovieLens dataset
    data = Dataset.load_builtin('ml-100k')
    df = pd.DataFrame(data.raw_ratings, columns=["user_id", "item_id", "rating", "timestamp"])

    # Normalize ratings
    df["rating"] = df["rating"] / df["rating"].max()

    # Map user and item IDs to consecutive integers
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {i: j for j, i in enumerate(df["item_id"].unique())}
    df["user_id"] = df["user_id"].map(user_map)
    df["item_id"] = df["item_id"].map(item_map)

    # Ensure there are no missing values after mapping
    df = df.dropna(subset=["user_id", "item_id"])

    # Create User-Item matrix
    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()
    user_item_matrix = np.zeros((num_users, num_items))
    for _, row in df.iterrows():
        user_item_matrix[int(row["user_id"]), int(row["item_id"])] = row["rating"]

    # Train-test split
    train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)

    return train_tensor, test_tensor, num_users, num_items