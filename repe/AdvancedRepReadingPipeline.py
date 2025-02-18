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

    def _get_string_variants(self, base_string):
        """Helper function to generate variants of a search string"""
        if base_string == "":
            return [""]
        variants = [
            base_string,
            "\n" + base_string,
            "\n\n" + base_string,
            " \n\n" + base_string,
            " " + base_string
        ]
        return variants
        
    def _process_string_search(self, inputs, searched_tokens_true=None, searched_tokens_false=None, labels=None):
        """Helper method to process string search and cut inputs based on labels"""
        if searched_tokens_true is None and searched_tokens_false is None:
            return inputs
    
        # Process both search strings
        searched_tokens_dict = {}
        for label, search_string in [(True, searched_tokens_true), (False, searched_tokens_false)]:
            if search_string is not None:
                if search_string == "":
                    searched_tokens_dict[label] = "<CUT!>"
                else:
                    variants = self._get_string_variants(search_string)
                    searched_tokens_dict[label] = []
                    for variant in variants:
                        tokenized = self.pipeline.tokenizer(variant, return_tensors="pt")["input_ids"]
                        searched_tokens_dict[label].append(tokenized[:, 1:])  # Remove BOS token
                        print(f"Searched tokens for label {label}, variant '{variant}':")
                        for t in tokenized[0, 1:]:
                            print(f"Token {t} -> {self.pipeline.tokenizer.decode(t)}")
    
        # Tokenize inputs
        train_inputs_tokens = self.pipeline.tokenizer(inputs, return_tensors="pt", padding=True)
        input_ids = train_inputs_tokens["input_ids"]
        
        # Cut inputs at found positions based on labels
        cut_inputs = []
        for idx, tokens in enumerate(input_ids):
            # Determine which search tokens to use based on label
            label_idx = idx // 2  # Integer division to get pair index
            label_pos = idx % 2   # Remainder to get position within pair
            if labels is None or len(labels) <= label_idx:
                searched_tokens_list = searched_tokens_dict.get(True, None)  # Default to True if no labels
            else:
                label = labels[label_idx][label_pos]
                searched_tokens_list = searched_tokens_dict.get(label, None)
            
            if searched_tokens_list is None:
                cut_inputs.append(self.pipeline.tokenizer.decode(tokens))
                continue
                
            if searched_tokens_list == "<CUT!>":
                # Find first non-padding token
                non_pad_tokens = tokens[tokens != self.pipeline.tokenizer.pad_token_id]
                if len(non_pad_tokens) == 0:
                    cut_inputs.append(self.pipeline.tokenizer.decode(tokens))
                    continue
                    
                # Calculate cut position relative to non-padded length
                non_pad_cut_pos = max(1, int(len(non_pad_tokens) * 0.3))
                
                # Find the actual position in the original tokens
                pad_count = (tokens == self.pipeline.tokenizer.pad_token_id).sum().item()
                cut_pos = pad_count + non_pad_cut_pos
                
                cut_tokens = tokens[:cut_pos]
                cut_text = self.pipeline.tokenizer.decode(cut_tokens)
                cut_inputs.append(cut_text)
                continue
    
            # Search for any of the variants
            found = False
            earliest_cut_pos = None
            
            for searched_tokens in searched_tokens_list:
                search_sequence = searched_tokens[0]
                sequence_length = len(search_sequence)
                
                for i in range(len(tokens) - sequence_length + 1):
                    window = tokens[i:i+sequence_length]
                    if (window == search_sequence).all():
                        cut_pos = i + sequence_length
                        if earliest_cut_pos is None or cut_pos < earliest_cut_pos:
                            earliest_cut_pos = cut_pos
                        found = True
                        
            if found and earliest_cut_pos is not None:
                cut_tokens = tokens[:earliest_cut_pos]
                cut_text = self.pipeline.tokenizer.decode(cut_tokens)
                cut_inputs.append(cut_text)
            else:
                cut_inputs.append(self.pipeline.tokenizer.decode(tokens))
    
        # Remove artifacts from decoded strings
        return [text.replace(self.pipeline.tokenizer.pad_token, "").replace(self.pipeline.tokenizer.bos_token,"") for text in cut_inputs]

    def get_directions(self, train_inputs, rep_token=-1, hidden_layers=-1,
                      n_difference=1, batch_size=8, train_labels=None,
                      direction_method='pca', direction_finder_kwargs={},
                      which_hidden_states=None, searched_tokens_true=None,
                      searched_tokens_false=None):
        
        processed_inputs = self._process_string_search(
            train_inputs, 
            searched_tokens_true, 
            searched_tokens_false, 
            train_labels
        )     
        
        return self.pipeline.get_directions(
            train_inputs=processed_inputs,
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            n_difference=n_difference, 
            batch_size=batch_size,
            train_labels=train_labels, 
            direction_method=direction_method)

    def rep_reading_test(self, test_inputs, rep_token=-1, hidden_layers=-1, rep_reader=None,
                        batch_size=32, searched_tokens_true=None, searched_tokens_false=None,
                        test_labels=None):
        """
        Process test data through the pipeline with optional string search based on labels
        """
        processed_inputs = self._process_string_search(
            test_inputs,
            searched_tokens_true,
            searched_tokens_false,
            test_labels
        )
        
        return self.pipeline(
            processed_inputs,
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            rep_reader=rep_reader,
            batch_size=batch_size)