from typing import List, Union, Optional, Tuple
from transformers import Pipeline
import torch
import numpy as np
from .rep_readers import DIRECTION_FINDERS, RepReader
from tqdm.notebook import tqdm  # Use tqdm notebook for better progress bars in Jupyter
from .rep_reading_pipeline import RepReadingPipeline

class AdvancedRepReaderWrapper:
    """
    Wraps RepReadingPipeline to add advanced token selection and direction finding.
    """
    def __init__(self, rep_reading_pipeline: RepReadingPipeline):
        self.pipeline = rep_reading_pipeline

    def get_directions(self, train_inputs: Union[str, List[str], List[List[str]]], hidden_layers: Union[List[int], int] = -1, n_difference: int = 1, batch_size: int = 8, train_labels: List[int] = None, direction_method: str = 'pca', direction_finder_kwargs: dict = {}, search_tokens: Union[str, List[str]] = None, sentence_selection: Tuple[int, int] = (0, 0), which_hidden_states: Optional[str] = None, **tokenizer_args):
        """
        Get concept directions using advanced token selection.

        Args:
            rep_reading_pipeline: An instance of RepReadingPipeline.
            search_tokens (str or list): Token or tokens to search for.
            sentence_selection (tuple): Tokens to select around the search token.
        """
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]

        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)
        all_layer_directions = {} # Store directions for each search token and layer

        if direction_finder.needs_hiddens:
            print("Extracting hidden states for search tokens...")
            for layer in tqdm(hidden_layers, desc="Processing layers"): # Layer-wise progress bar
                all_token_directions_for_layer = []
                if isinstance(search_tokens, str) or isinstance(search_tokens, list):
                    if isinstance(search_tokens, str):
                        search_tokens_list = [search_tokens] # Handle single string input
                    else:
                        search_tokens_list = search_tokens

                    for tokens_to_search in tqdm(search_tokens_list, desc="Processing search tokens", leave=False): # Token-wise nested progress bar
                        directions_for_token = self.pipeline.get_directions( # Use original pipeline for direction extraction
                            train_inputs,
                            hidden_layers=[layer], # Extract directions for current layer only
                            n_difference=n_difference,
                            train_labels=train_labels,
                            direction_method=direction_method,
                            direction_finder_kwargs=direction_finder_kwargs,
                            rep_token = -1, # Use dummy rep_token, actual token selection happens in _forward
                            which_hidden_states=which_hidden_states,
                            **tokenizer_args,
                            component_index=0, # Pass component_index
                            search_tokens = tokens_to_search, # Pass current search token
                            sentence_selection = sentence_selection # Pass sentence_selection
                        )
                        all_token_directions_for_layer.append(directions_for_token.directions[layer]) # Store directions

                    # Merge directions for multi-token search
                    if all_token_directions_for_layer:
                        merged_directions = np.mean(np.array(all_token_directions_for_layer), axis=0) # Average directions
                        all_layer_directions[layer] = merged_directions
                    else: # Handle cases where no tokens are found, use random direction as fallback
                        print(f"Warning: Search tokens '{search_tokens_list}' not found in training inputs for layer {layer}. Using random direction.")
                        continue

        direction_finder.directions = all_layer_directions
        direction_finder.n_components = direction_finder.directions[hidden_layers[0]].shape[0] if direction_finder.directions else 0 # Update n_components
        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
                hidden_states, train_labels, hidden_layers) # Use raw hidden states for signs, as in original pipeline

        return direction_finder


    def __call__(self, inputs: Union[str, List[str], List[List[str]]], batch_size=8, **kwargs):
        """
        Override __call__ to use _forward with search_tokens and sentence_selection.
        """
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)

        # Use original pipeline's preprocess and postprocess methods
        model_inputs = self.preprocess(inputs, **preprocess_params)
        
        forward_params_updated = {**forward_params, 'pad_token_id': self.tokenizer.pad_token_id} # Ensure pad_token_id is passed
        
        if "search_tokens" in forward_params and "sentence_selection" in forward_params:
            # Use overridden _forward with token selection logic
            outputs = self._forward(model_inputs, **forward_params_updated)
        else:
            # Fallback to original _forward if search_tokens or sentence_selection are not provided
            outputs = super()._forward(model_inputs, **forward_params_updated)

        model_outputs = self.postprocess(outputs, **postprocess_params)
        return model_outputs