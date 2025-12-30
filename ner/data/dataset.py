import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, encodings): 
        self.encodings = encodings

    def __len__(self): 
        return len(self.encodings)
    
    def __getitem__(self, idx): 
        return {k: torch.tensor(v) for k,v in self.encodings[idx].items()}