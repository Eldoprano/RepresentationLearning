import torch
import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Literal
from tqdm import tqdm
from .rep_readers import RepReader

class PatternRepReader(RepReader):
    """Pattern-based representation reader that finds relevant tokens using regex patterns"""
    
    def __init__(self, 
                 target_pattern: str,
                 pattern_token_selection: Literal["first", "last", "all"] = "last",
                 sentence_context: Tuple[int, int] = (0, 0)):
        super().__init__()
        self.target_pattern = target_pattern
        self.pattern_token_selection = pattern_token_selection
        self.before_sentences, self.after_sentences = sentence_context
        self.directions = {}
        self.direction_signs = {}

    def _find_matches(self, text):
        """Find positive and negative pattern matches in text"""
        # Get text between start and end patterns
        pos_pattern = f"{self.start_pattern}.*?{self.end_pattern}"
        pos_matches = re.finditer(pos_pattern, text, re.DOTALL)
        pos_spans = [m.span() for m in pos_matches]
        
        neg_spans = []
        if self.negative_pattern:
            neg_matches = re.finditer(self.negative_pattern, text, re.DOTALL)
            neg_spans = [m.span() for m in neg_matches]
            
        return pos_spans, neg_spans

    def get_rep_directions(self, texts, reps, layers):
        """Get directions distinguishing between positive and negative pattern matches"""
        all_pos_reps = []
        all_neg_reps = []
        
        # Collect positive and negative examples for each text
        for text, text_reps in zip(texts, reps):
            pos_spans, neg_spans = self._find_matches(text)
            
            if not pos_spans and not neg_spans:
                continue
                
            # Get representations for matched spans
            for s, e in pos_spans:
                all_pos_reps.append([r[self.rep_token] for r in text_reps])
                
            for s, e in neg_spans:
                all_neg_reps.append([r[self.rep_token] for r in text_reps])
                
        # Skip if not enough examples
        if len(all_pos_reps) < 1 or (self.negative_pattern and len(all_neg_reps) < 1):
            return None
            
        # Calculate directions
        directions = {}
        signs = {}
        
        for layer in layers:
            pos_vecs = [rep[layer] for rep in all_pos_reps]
            
            if len(all_neg_reps) > 0:
                neg_vecs = [rep[layer] for rep in all_neg_reps]
                direction = np.mean(pos_vecs, axis=0) - np.mean(neg_vecs, axis=0)
            else:
                direction = np.mean(pos_vecs, axis=0)
                
            # Normalize
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                
            directions[layer] = direction
            signs[layer] = 1.0  # Direction from negative to positive
            
        self.directions = directions
        self.direction_signs = signs
        
        return directions

    def get_directions(self, 
                      texts: List[str],
                      rep_token: int = -1,
                      hidden_layers: List[int] = [-1],
                      train_labels: Optional[List[int]] = None,
                      direction_method: str = "pca",
                      batch_size: int = 32,
                      **kwargs) -> Dict[int, torch.Tensor]:
        """Get directions from pattern matches using existing pipeline"""
        
        # For each text, find pattern matches and collect text+positions+labels
        all_matches = []  # List of (text, position, label) tuples
        
        for i, text in tqdm(enumerate(texts), desc="Finding patterns", total=len(texts)):
            matches = list(re.finditer(re.escape(self.target_pattern), text))
            if not matches:
                continue

            for match in matches:
                pattern_start = len(self.tokenizer.encode(text[:match.start()])) - 1
                pattern_length = len(self.tokenizer.encode(self.target_pattern))
                
                # Select token positions based on configuration  
                if self.pattern_token_selection == "first":
                    positions = [pattern_start]
                elif self.pattern_token_selection == "last":
                    positions = [pattern_start + pattern_length - 1] 
                else:  # "all"
                    positions = list(range(pattern_start, pattern_start + pattern_length))
                
                # Add each position as a separate match
                for pos in positions:
                    all_matches.append((
                        text,
                        pos,  # Single integer position
                        train_labels[i] if train_labels else None
                    ))

        if not all_matches:
            raise ValueError("No pattern matches found in texts")

        directions = {}
        # Process in batches to avoid memory issues
        for batch_start in tqdm(range(0, len(all_matches), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(all_matches))
            batch = all_matches[batch_start:batch_end]
            
            # Unzip batch
            batch_texts = [item[0] for item in batch]
            batch_positions = [item[1] for item in batch]  # Each position is a single int
            batch_labels = [item[2] for item in batch] if train_labels else None

            # Get hidden states for all layers at once
            states = super().get_hidden_states(
                texts=batch_texts,
                rep_token=batch_positions,  # List of individual positions
                hidden_layers=hidden_layers,
                batch_size=batch_size
            )

            # Process each layer
            for layer in hidden_layers:
                layer_states = states[layer]
                
                if layer not in directions:
                    directions[layer] = []

                if batch_labels and direction_method == "contrast":
                    # Use supervised contrast method if labels available
                    direction = super()._get_contrast_direction(
                        states=layer_states,
                        labels=batch_labels,
                        **kwargs
                    )
                else:
                    # Use PCA for this batch
                    direction = self._get_pca_direction(layer_states)
                
                directions[layer].append(direction)

        # Average directions for each layer
        final_directions = {}
        for layer in hidden_layers:
            final_directions[layer] = torch.stack(directions[layer]).mean(dim=0)
            
            # Store direction signs for this layer
            self.direction_signs[layer] = self._get_direction_signs(
                final_directions[layer],
                train_labels=train_labels
            ) if train_labels else torch.ones(1, device=final_directions[layer].device)

        return final_directions

    def _get_pca_direction(self, states: torch.Tensor) -> torch.Tensor:
        """Get primary PCA direction from states"""
        # Center the data
        states_centered = states - states.mean(dim=0)
        # Get covariance matrix
        cov = states_centered.T @ states_centered
        # Get primary eigenvector
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        return eigenvectors[:, -1]  # Return direction of maximum variance

    def compare_directions(self, other_reader: 'PatternRepReader', layer: int) -> float:
        """Compare similarity between LAT directions at specified layer"""
        direction1 = self.directions[layer]
        direction2 = other_reader.directions[layer]
        
        # Normalize directions
        direction1 = direction1 / torch.norm(direction1)
        direction2 = direction2 / torch.norm(direction2)
        
        # Compute absolute cosine similarity 
        similarity = abs(torch.dot(direction1, direction2).item())
        return similarity
