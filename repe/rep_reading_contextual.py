import torch
import numpy as np
from transformers import Pipeline
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm
from .rep_reading_pipeline import RepReadingPipeline

class ContextualRepReadingPipeline(RepReadingPipeline):
    """Extended RepReadingPipeline that allows for contextual token-based representation reading"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _find_token_sequence(self, tokens: torch.Tensor, search_sequence: torch.Tensor) -> List[int]:
        """Find positions of a token sequence in tokenized input"""
        # Convert tensors to lists for easier manipulation
        tokens_list = tokens.tolist()
        search_list = search_sequence.tolist() 
        
        # Find the token sequence
        seq_len = len(search_list)
        for i in range(len(tokens_list) - seq_len + 1):
            if tokens_list[i:i + seq_len] == search_list:
                return list(range(i, i + seq_len))
                
        raise ValueError("Token sequence not found")

    def _get_context_slice(self, 
                          tokens: torch.Tensor,
                          seq_pos: List[int],
                          context: Tuple[int, int]) -> List[int]:
        """Get token positions for the requested context window
        
        Args:
            context: Tuple of (tokens_before, tokens_after). Use -1 to exclude search sequence.
        """
        # Just return sequence positions if no context requested
        if context == (0, 0):
            return seq_pos
            
        seq_start = seq_pos[0] 
        seq_end = seq_pos[-1]
        
        # Calculate window boundaries
        window_start = max(0, seq_start - context[0]) if context[0] >= 0 else seq_start
        window_end = min(len(tokens), seq_end + context[1]) if context[1] >= 0 else seq_end
        
        # Get context positions
        context_pos = list(range(window_start, window_end + 1))
        
        # Remove search sequence positions if requested
        if context[0] == -1 or context[1] == -1:
            context_pos = [pos for pos in context_pos if pos not in seq_pos]
            
        # Print selected tokens
        selected_tokens = self.tokenizer.decode(tokens[context_pos])
        print(f"\nSelected context: '{selected_tokens}'")
        print(f"Context token positions: {context_pos}")
            
        return context_pos

    def _merge_directions(self, directions_list: List[torch.Tensor]) -> torch.Tensor:
        """Merge multiple directions into one by moving to CPU first"""
        directions_cpu = [d.cpu() for d in directions_list]
        merged = torch.mean(torch.stack(directions_cpu, dim=0), dim=0)
        return merged

    def get_directions(self, 
                      dataset: List[torch.Tensor],
                      search_sequence: Optional[torch.Tensor] = None,
                      context: Tuple[int, int] = (0, 0),
                      hidden_layers: List[int] = None,
                      train_labels: Optional[List[int]] = None,
                      **kwargs) -> Dict:
        """Get representation directions based on token sequence search with context"""
        if search_sequence is None:
            return super().get_directions(dataset, hidden_layers=hidden_layers, 
                                       train_labels=train_labels, **kwargs)
        
        # Get positions for token sequence in each input
        print("\nProcessing texts for direction extraction:")
        all_positions = []
        for tokens in tqdm(dataset):
            seq_pos = self._find_token_sequence(tokens, search_sequence) 
            context_pos = self._get_context_slice(tokens, seq_pos, context)
            all_positions.append(context_pos)
            
        # Get directions for each text position first, then merge by layer
        all_directions = {layer: [] for layer in hidden_layers}
        
        for text_idx, positions in enumerate(tqdm(all_positions)):
            # Get directions for all positions in this text at once
            for position in positions:
                dir_result = super().get_directions(
                    [dataset[text_idx]], 
                    rep_token=position,
                    hidden_layers=hidden_layers,
                    train_labels=None if train_labels is None else [train_labels[text_idx]],
                    **kwargs
                )
            
                # Store directions by layer 
                for layer in hidden_layers:
                    all_directions[layer].append(dir_result[layer].cpu())
                del dir_result
                torch.cuda.empty_cache()
                
        # Merge directions for each layer
        merged_directions = {}
        for layer in hidden_layers:
            merged_directions[layer] = self._merge_directions(all_directions[layer])
            
        return merged_directions

    def __call__(self,
                 dataset: List[torch.Tensor],
                 search_sequence: Optional[torch.Tensor] = None, 
                 context: Tuple[int, int] = (0, 0),
                 **kwargs):
        """Pipeline call with token sequence search support"""
        # If no search sequence, use parent implementation
        if search_sequence is None:
            return super().__call__(dataset, **kwargs)
            
        # Get positions for token sequence
        all_positions = []
        for tokens in dataset:
            seq_pos = self._find_token_sequence(tokens, search_sequence)
            context_pos = self._get_context_slice(tokens, seq_pos, context)
            all_positions.append(context_pos)
        
        # Call parent's __call__ for each position and merge results
        all_results = []
        for text_idx, positions in enumerate(all_positions):
            text_results = []
            for pos in positions:
                result = super().__call__([dataset[text_idx]], rep_token=pos, **kwargs)
                text_results.append(result[0])
            
            # Merge results for this text
            merged = {}
            for layer in text_results[0].keys():
                merged[layer] = torch.mean(torch.stack(
                    [r[layer] for r in text_results]
                ), dim=0)
            all_results.append(merged)
            
        return all_results
