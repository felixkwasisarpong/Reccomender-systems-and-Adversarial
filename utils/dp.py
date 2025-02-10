# utils/dp.py
import torch

def add_dp_noise(embedding, epsilon=0.1):
    """Add noise to embeddings to simulate differential privacy."""
    noise = torch.normal(mean=0, std=1/epsilon, size=embedding.shape)
    return embedding + noise
