import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def plot_rating_distribution(h5_path):
    with h5py.File(h5_path, 'r') as file:
        ratings = file['ratings']['rating'][:].astype(np.float32)

    rating_counts = Counter(ratings)
    sorted_ratings = sorted(rating_counts.items())

    print("📊 Rating Distribution:")
    for rating, count in sorted_ratings:
        print(f"Rating {rating:.1f}: {count} ({(count / len(ratings)) * 100:.2f}%)")

    # Optional: Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[r[0] for r in sorted_ratings], y=[r[1] for r in sorted_ratings], palette="muted")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks([1.0, 2.0, 3.0, 4.0, 5.0])
    plt.show()

# Example usage:
plot_rating_distribution('//Users/Apple/Documents/assignements/Thesis/netflix_data/netflix_data.hdf5')
