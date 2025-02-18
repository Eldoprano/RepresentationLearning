import torch
import numpy as np
from tqdm.notebook import tqdm  # Use tqdm notebook for better progress bars in Jupyter
from typing import List, Union, Optional


class StringSearchRepReader:
    """
    Wraps RepReadingPipeline to add string searching for hidden state extraction.
    """
    def __init__(self, rep_reading_pipeline):
        self.pipeline = rep_reading_pipeline
        
    def get_directions(            self, 
            train_inputs: Union[str, List[str], List[List[str]]], 
            rep_token: Union[str, int]=-1, 
            hidden_layers: Union[str, int]=-1,
            n_difference: int = 1,
            batch_size: int = 8, 
            train_labels: List[int] = None,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str]=None,
            searched_tokens=None):
        
        if searched_tokens is None:
            return self.pipeline.get_directions(train_inputs, rep_token, hidden_layers, 
                                                n_difference, train_labels, direction_method, 
                                                batch_size)

        # Tokenize search string and remove BOS token
        searched_tokens = self.pipeline.tokenizer(searched_tokens, return_tensors="pt")["input_ids"]
        searched_tokens = searched_tokens[:, 1:]
        # Show user which tokens will be searched
        print("Searched tokens:")
        for t in searched_tokens[0]:
            print(f"Token {t} -> {self.pipeline.tokenizer.decode(t)}")
        
        # Tokenize inputs
        train_inputs_tokens = self.pipeline.tokenizer(train_inputs, return_tensors="pt", padding=True)
        input_ids = train_inputs_tokens["input_ids"]
        
        # Cut inputs at found positions
        cut_inputs = []
        for tokens in input_ids:
            for i in range(len(tokens) - len(searched_tokens[0]) + 1):
                if torch.all(tokens[i:i+len(searched_tokens[0])] == searched_tokens[0]):
                    cut_tokens = tokens[:i+len(searched_tokens[0])]
                    cut_text = self.pipeline.tokenizer.decode(cut_tokens)
                    cut_inputs.append(cut_text)
                    break
        
        # Remove artifacts from decoded strings
        cut_inputs = [text.replace(self.pipeline.tokenizer.pad_token, "").replace(self.pipeline.tokenizer.bos_token,"") for text in cut_inputs]
  
        # Call RepReadingPipeline
        return self.pipeline.get_directions(
            train_inputs=cut_inputs,
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            batch_size=batch_size,
            train_labels=train_labels, 
            direction_method=direction_method)