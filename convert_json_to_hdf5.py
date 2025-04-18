import h5py
import json
import os
import glob
from collections import defaultdict


def verify_hdf5_file(output_file):
    with h5py.File(output_file, 'r') as hf:
        user_ids = hf['user_ids'][:]
        business_ids = hf['business_ids'][:]
        ratings = hf['ratings'][:]

        print("User IDs:", user_ids)
        print("Business IDs:", business_ids)
        print("Ratings:", ratings)

# Example usage
output_file = "yelp_data.h5"
verify_hdf5_file(output_file)
