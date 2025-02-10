# main.py
from data.load_movielens import load_movielens
from training.train import Trainer

def main():
    # Load data
    train_tensor, _, num_users, num_items = load_movielens()
    
    # Initialize and train models
    trainer = Trainer(train_tensor, num_users, num_items)
    trainer.train_recommender(epochs=5)
    trainer.train_inversion_model(epochs=5, epsilon=0.1)
    trainer.train_gan(epochs=10)

if __name__ == "__main__":
    main()
