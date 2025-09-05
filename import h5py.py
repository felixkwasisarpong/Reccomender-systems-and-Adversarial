import h5py
import numpy as np

h5_path = "predicted_data/custom_strong.hdf5"   # adjust to your file

with h5py.File(h5_path, "r") as f:
    print("Keys in file:", list(f.keys()))  # should show ['predictions']
    preds = f["predictions"][:]             # structured array
    print("dtype:", preds.dtype)

    # Peek first 5 rows
    print("\nSample rows:")
    print(preds[:5])

    # Extract fields
    user_ids   = preds["user_id"][:10]
    movie_ids  = preds["movie_id"][:10]
    ratings    = preds["pred_rating"][:10]
    genders    = preds["gender"][:10]
    ages       = preds["age"][:10]
    occs       = preds["occupation"][:10]
    genres     = preds["genre"][:10]

    print("\nSample raw values:")
    print("user_id:", user_ids)
    print("movie_id:", movie_ids)
    print("pred_rating:", ratings)
    print("gender:", genders)
    print("age:", ages)
    print("occupation:", occs)
    print("genre[0]:", genres[0])  # show one row's genre vector

    # Quick sanity ranges
    print("\nRanges:")
    print("user_id range:", np.min(user_ids), np.max(user_ids))
    print("movie_id range:", np.min(movie_ids), np.max(movie_ids))
    print("pred_rating range:", np.min(ratings), np.max(ratings))
    print("age range:", np.min(ages), np.max(ages))
    print("occupation range:", np.min(occs), np.max(occs))
    print("gender distribution:", np.unique(genders, return_counts=True))