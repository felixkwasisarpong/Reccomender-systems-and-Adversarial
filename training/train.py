# training/trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
from model.recommender import Recommender
from model.inversion_model import InversionModel
from model.gan import Generator, Discriminator
from utils.dp import add_dp_noise

class Trainer:
    def __init__(self, train_data, num_users, num_items, embedding_dim=64):
        self.train_data = train_data
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Initialize models
        self.recommender = Recommender(num_users, num_items, embedding_dim)
        self.inversion_model = InversionModel(embedding_dim, num_items)
        self.generator = Generator(embedding_dim, num_items)
        self.discriminator = Discriminator(num_items)

        # Optimizers
        self.optimizer_recommender = optim.Adam(self.recommender.parameters(), lr=0.001)
        self.optimizer_inversion = optim.Adam(self.inversion_model.parameters(), lr=0.001)
        self.optimizer_gan_gen = optim.Adam(self.generator.parameters(), lr=0.0001)
        self.optimizer_gan_disc = optim.Adam(self.discriminator.parameters(), lr=0.0001)

        # Loss function
        self.criterion = nn.MSELoss()

    def train_recommender(self, epochs=5):
        for epoch in range(epochs):
            for user_idx in range(self.train_data.shape[0]):  # Iterate over each user
                user_ratings = self.train_data[user_idx]
                user_ids = torch.full_like(user_ratings, user_idx, dtype=torch.long)  # Ensure user_ids are Long
                ratings_pred = self.recommender(user_ids)
                loss = self.criterion(ratings_pred, user_ratings)
                self.optimizer_recommender.zero_grad()
                loss.backward()
                self.optimizer_recommender.step()
            print(f"Recommender - Epoch {epoch+1}, Loss: {loss.item():.4f}")



    def train_inversion_model(self, epochs=5, epsilon=0.1):
        for epoch in range(epochs):
            for user_idx in range(self.num_users):
                if user_idx >= self.train_data.shape[0]:  # Ensure within bounds
                    continue
                dp_embedding = add_dp_noise(self.recommender.user_embed.weight[user_idx], epsilon)
                reconstructed = self.inversion_model(dp_embedding)
                loss = self.criterion(reconstructed, torch.tensor(self.train_data[user_idx], dtype=torch.float32))
                
                self.optimizer_inversion.zero_grad()
                loss.backward()
                self.optimizer_inversion.step()
            print(f"Inversion Model - Epoch {epoch+1}, Loss: {loss.item():.4f}")


    def train_gan(self, epochs=10):
        for epoch in range(epochs):
            for user_idx in range(self.num_users):
                if user_idx >= self.train_data.shape[0]:  # Ensure within bounds
                    continue
                dp_embedding = add_dp_noise(self.recommender.user_embed.weight[user_idx])

                # Train Discriminator
                self.optimizer_gan_disc.zero_grad()
                real_data = torch.tensor(self.train_data[user_idx], dtype=torch.float32).unsqueeze(0)
                fake_data = self.generator(self.inversion_model(dp_embedding)).detach()

                real_loss = self.criterion(self.discriminator(real_data), torch.ones(1, 1))
                fake_loss = self.criterion(self.discriminator(fake_data), torch.zeros(1, 1))
                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                self.optimizer_gan_disc.step()

                # Train Generator
                self.optimizer_gan_gen.zero_grad()
                fake_data = self.generator(self.inversion_model(dp_embedding))
                gen_loss = self.criterion(self.discriminator(fake_data), torch.ones(1, 1))
                gen_loss.backward()
                self.optimizer_gan_gen.step()

            print(f"GAN - Epoch {epoch+1}, D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")

