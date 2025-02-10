# models/inversion_model.py
import torch
import torch.nn as nn

class InversionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InversionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Output user preferences
