import os
import tarfile
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, List
from dataloader.NetflixPredictor import NetflixPredictor
from dataloader.NetflixDataset import NetflixDataset
import numpy as np

class NetflixPredictor:
    def __init__(self, dataset: NetflixDataset):
        self.dataset = dataset
        # Create reverse mappings
        self.int_to_user_id = {v: k for k, v in dataset.user_id_to_int.items()}
        self.int_to_movie_id = {v: k for k, v in dataset.movie_id_to_int.items()}
    
    def load_qualifying_data(self, file_path: str) -> Dict[int, List[Dict]]:
        """Load qualifying data from file"""
        qualifying_data = {}
        current_movie_id = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.endswith(':'):
                    current_movie_id = int(line[:-1])
                    qualifying_data[current_movie_id] = []
                elif current_movie_id is not None:
                    customer_id, date = line.split(',')
                    qualifying_data[current_movie_id].append({
                        'customer_id': int(customer_id),
                        'date': date
                    })
        
        return qualifying_data

    def generate_predictions(self, model, qualifying_data: Dict[int, List[Dict]], output_file: str):
        """Generate predictions in Netflix submission format"""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for movie_id, entries in qualifying_data.items():
                predictions.append(f"{movie_id}:")
                
                batch_user_ids = []
                batch_movie_ids = []
                valid_entries = []
                
                for entry in entries:
                    customer_id = entry['customer_id']
                    
                    # Check if user/movie exists in training data
                    if (customer_id in self.dataset.user_id_to_int and 
                        movie_id in self.dataset.movie_id_to_int):
                        batch_user_ids.append(self.dataset.user_id_to_int[customer_id])
                        batch_movie_ids.append(self.dataset.movie_id_to_int[movie_id])
                        valid_entries.append(entry)
                    else:
                        # Fallback to global average (3.5) for unknown user/movie
                        predictions.append("3.5")
                
                if batch_user_ids:
                    # Batch predict for efficiency
                    user_tensor = torch.tensor(batch_user_ids, dtype=torch.long)
                    movie_tensor = torch.tensor(batch_movie_ids, dtype=torch.long)
                    batch_preds = model(user_tensor, movie_tensor).cpu().numpy()
                    
                    # Clip predictions to valid range [1, 5]
                    batch_preds = np.clip(batch_preds, 1.0, 5.0)
                    
                    # Format predictions to one decimal place
                    for pred in batch_preds:
                        predictions.append(f"{pred:.1f}")
        
        # Save predictions to file
        with open(output_file, 'w') as f:
            f.write("\n".join(predictions))
        
        return predictions