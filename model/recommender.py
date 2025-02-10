# models/recommender.py
import torch
import torch.nn as nn

class Recommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(Recommender, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_items)

    def forward(self, user_ids):
        user_vec = self.user_embed(user_ids)
        return self.fc(user_vec)
