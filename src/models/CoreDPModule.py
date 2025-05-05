import torch
import torch.nn as nn
from .BaseModel import BaseModel

class CoreDPModule(nn.Module):
    def __init__(self, user_embedding, item_embedding, dropout, fc):
        super().__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.dropout = nn.Dropout(dropout)  # <- wrap float in Dropout layer
        self.fc = fc
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        x = torch.cat([user_embed, item_embed], dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze()
