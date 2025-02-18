from typing import List, Union, Optional, Tuple
from transformers import Pipeline
import torch
import numpy as np
from .rep_readers import DIRECTION_FINDERS, RepReader

from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModel, AutoModelForCausalLM
from .rep_reading_pipeline import RepReadingPipeline
from .rep_control_pipeline import RepControlPipeline

PIPELINE_REGISTRY.register_pipeline(
    "rep-reading",
    pipeline_class=RepReadingPipeline,
    pt_model=AutoModel,
)

PIPELINE_REGISTRY.register_pipeline(
    "rep-control",
    pipeline_class=RepControlPipeline,
    pt_model=AutoModelForCausalLM,
)

class AdvancedRepReadingPipeline(RepReadingPipeline):
    """
    RepReadingPipeline with advanced token selection for direction finding.
    """

    def _sanitize_parameters(self,
                             rep_reader: RepReader = None,
                             search_tokens: Union[str, List[str]] = None,
                             sentence_selection: Tuple[int, int] = (0, 0),
                             hidden_layers: Union[List[int], int] = -1,
                             component_index: int = 0,
                             which_hidden_states: Optional[str] = None,
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}

        forward_params['search_tokens'] = search_tokens
        forward_params['sentence_selection'] = sentence_selection

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]

        assert rep_reader is None or len(rep_reader.directions) == len(hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states

        return preprocess_params, forward_params, postprocess_params


    def _forward(self, model_inputs, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None, search_tokens=None, sentence_selection=(0,0), pad_token_id=None):
        """Override _forward to use search_tokens and sentence_selection."""

        all_hidden_states = {}
        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input
            outputs =  self.model(**model_inputs, output_hidden_states=True)

            for layer in hidden_layers:
                layer_hidden_states = outputs['hidden_states'][layer]
                token_positions = []

                if isinstance(search_tokens, str) or isinstance(search_tokens, list):
                    if isinstance(search_tokens, str):
                        search_tokens_list = [search_tokens] # Handle single string input
                    else:
                        search_tokens_list = search_tokens

                    for tokens_to_search in search_tokens_list:
                        found_token_ids = self.tokenizer.encode(tokens_to_search, add_special_tokens=False)
                        for batch_idx in range(layer_hidden_states.shape[0]):
                            input_ids_batch = model_inputs['input_ids'][batch_idx]
                            start_positions = (input_ids_batch == found_token_ids[0]).nonzero(as_tuple=True)[0]

                            current_positions = []
                            for start_pos in start_positions:
                                match = True
                                for offset, token_id in enumerate(found_token_ids[1:]):
                                    if start_pos + offset + 1 >= input_ids_batch.shape[0] or input_ids_batch[start_pos + offset + 1] != token_id:
                                        match = False
                                        break
                                if match:
                                    current_positions.append(start_pos.item())

                            if len(current_positions) > 0:
                                token_positions.extend(current_positions) # Add all instances found in the batch
                else: # Fallback to original rep_token logic if search_tokens is None or not a valid type
                    rep_token = -1 # else default to -1
                    token_positions = [rep_token] * layer_hidden_states.shape[0]


                selected_hidden_states_list = []
                for batch_idx in range(layer_hidden_states.shape[0]):
                    start_offset, end_offset = sentence_selection
                    merged_hiddens = []
                    found_search_tokens = False # Flag to track if search tokens were found
                    for pos_index, pos in enumerate(token_positions): # Loop through positions
                        if pos != -1:
                            found_search_tokens = True # Set flag to True because tokens were found
                            start_pos = max(0, pos + start_offset)
                            # Corrected line: use tokens_to_search instead of search_tokens_list[pos_index] which can cause IndexError
                            end_pos = min(layer_hidden_states.shape[1], pos + len(self.tokenizer.encode(tokens_to_search, add_special_tokens=False)) + end_offset) 
                            selected_hiddens = layer_hidden_states[batch_idx, start_pos:end_pos, :]
                            if selected_hiddens.numel() > 0:
                                merged_hiddens.append(selected_hiddens.mean(dim=0)) # Average if multiple tokens selected
                    if merged_hiddens:
                        selected_hidden_states_list.append(torch.stack(merged_hiddens).mean(dim=0)) # Average over all found instances
                    else: # Handle cases where search_tokens are not found in the batch
                        print(f"Search tokens: {search_tokens_list} not found in input, using last token instead.") # Print warning
                        selected_hidden_states_list.append(layer_hidden_states[batch_idx, -1, :]) # Fallback to last token


                selected_hidden_states = torch.stack(selected_hidden_states_list)

                if selected_hidden_states.dtype == torch.bfloat16:
                    selected_hidden_states = selected_hidden_states.float()
                all_hidden_states[layer] = selected_hidden_states.detach()

        return all_hidden_states


    def get_directions(self, train_inputs: Union[str, List[str], List[List[str]]], hidden_layers: Union[str, int] = -1, n_difference: int = 1, batch_size: int = 8, train_labels: List[int] = None, direction_method: str = 'pca', direction_finder_kwargs: dict = {}, search_tokens: Union[str, List[str]] = None, sentence_selection: Tuple[int, int] = (0, 0), which_hidden_states: Optional[str] = None, **tokenizer_args):
        """
        Override get_directions to use search_tokens and sentence_selection.
        """

        if not isinstance(hidden_layers, list):
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]

        self._validate_params(n_difference, direction_method)

        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            hidden_states = self._batched_string_to_hiddens(train_inputs, hidden_layers, batch_size, which_hidden_states, search_tokens=search_tokens, sentence_selection=sentence_selection, **tokenizer_args)

            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]

        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
            hidden_states, train_labels, hidden_layers)

        return direction_finder

    def _batched_string_to_hiddens(self, train_inputs, hidden_layers, batch_size, which_hidden_states, search_tokens, sentence_selection, **tokenizer_args):
        # Wrapper method to get a dictionary hidden states from a list of strings
        hidden_states_outputs = self(train_inputs, hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, which_hidden_states=which_hidden_states, search_tokens=search_tokens, sentence_selection=sentence_selection, **tokenizer_args)
        hidden_states = {layer: [] for layer in hidden_states_batch}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
        return {k: np.vstack(v) for k, v in hidden_states.items()}