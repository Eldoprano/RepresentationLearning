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
        
    def _process_string_search(self, inputs, searched_tokens):
        """Helper method to process string search and cut inputs"""
        if searched_tokens is None:
            return inputs

        # Tokenize search string and remove BOS token
        searched_tokens = self.pipeline.tokenizer(searched_tokens, return_tensors="pt")["input_ids"]
        searched_tokens = searched_tokens[:, 1:]
        # Show user which tokens will be searched
        print("Searched tokens:")
        for t in searched_tokens[0]:
            print(f"Token {t} -> {self.pipeline.tokenizer.decode(t)}")
        
        # Tokenize inputs
        train_inputs_tokens = self.pipeline.tokenizer(inputs, return_tensors="pt", padding=True)
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
        return [text.replace(self.pipeline.tokenizer.pad_token, "").replace(self.pipeline.tokenizer.bos_token,"") for text in cut_inputs]

    def get_directions(self, train_inputs, rep_token=-1, hidden_layers=-1,
                      n_difference=1, batch_size=8, train_labels=None,
                      direction_method='pca', direction_finder_kwargs={},
                      which_hidden_states=None, searched_tokens=None):
        
        processed_inputs = self._process_string_search(train_inputs, searched_tokens)
        
        return self.pipeline.get_directions(
            train_inputs=processed_inputs,
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            batch_size=batch_size,
            train_labels=train_labels, 
            direction_method=direction_method)

    def rep_reading_test(self, test_inputs, rep_token=-1, hidden_layers=-1, rep_reader=None,
                        batch_size=32, searched_tokens=None):
        """
        Process test data through the pipeline with optional string search
        """
        processed_inputs = self._process_string_search(test_inputs, searched_tokens)
        
        return self.pipeline(
            processed_inputs,
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            rep_reader=rep_reader,
            batch_size=batch_size)