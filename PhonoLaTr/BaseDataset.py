import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

class BaseDataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 max_input_length = 180,
                 max_output_length = 128,
                 truncation=True):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.truncation = truncation

        self.init_storage()
    

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ method has not implemented")
    
    def init_storage(self):
        self.feature = ["input_ids", "src_attention_mask", "label_ids", "label_attention_mask"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

    def data_processing(self):
        raise NotImplementedError