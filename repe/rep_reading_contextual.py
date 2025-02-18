import torch
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

class ContextualRepReader:
    """Wrapper for RepReadingPipeline that adds contextual token-based reading"""

    def __init__(self, pipeline):
        """Initialize with an existing RepReadingPipeline instance"""
        self.pipeline = pipeline
        self.tokenizer = pipeline.tokenizer

    def _find_token_sequence(self, tokens: torch.Tensor, search_sequence: torch.Tensor) -> List[int]:
        """Find positions of a token sequence in tokenized input"""
        tokens_list = tokens.tolist()
        search_list = search_sequence.tolist()

        seq_len = len(search_list)
        for i in range(len(tokens_list) - seq_len + 1):
            if tokens_list[i:i + seq_len] == search_list:
                print(f"_find_token_sequence: Token sequence found at positions: {list(range(i, i + seq_len))}")
                return list(range(i, i + seq_len))

        raise ValueError("Token sequence not found")

    def _get_context_slice(self, tokens: torch.Tensor,
                          seq_pos: List[int],
                          context: Tuple[int, int]) -> List[int]:
        """Get token positions for the requested context window"""
        if context == (0, 0):
            print("_get_context_slice: Context is (0, 0), returning sequence positions.")
            return seq_pos

        seq_start = seq_pos[0]
        seq_end = seq_pos[-1]

        window_start = max(0, seq_start - context[0]) if context[0] >= 0 else seq_start
        window_end = min(len(tokens), seq_end + context[1]) if context[1] >= 0 else seq_end

        context_pos = list(range(window_start, window_end + 1))

        if context[0] == -1 or context[1] == -1:
            context_pos = [pos for pos in context_pos if pos not in seq_pos]

        print(f"_get_context_slice: Context slice positions: {context_pos}")
        return context_pos

    def get_readings(self,
                    dataset: List[torch.Tensor],
                    search_sequence: Optional[torch.Tensor] = None,
                    context: Tuple[int, int] = (0, 0),
                    **kwargs) -> List[Dict]:
        """Get representation readings with token sequence search support"""
        if search_sequence is None:
            return self.pipeline(dataset, **kwargs)

        # Get positions for each text
        all_positions = []
        for tokens in dataset:
            seq_pos = self._find_token_sequence(tokens, search_sequence)
            context_pos = self._get_context_slice(tokens, seq_pos, context)
            all_positions.append(context_pos)

        # Get readings for all positions in each text
        all_results = []
        for text_idx, positions in enumerate(all_positions):
            text_results = []
            for pos in positions:
                result = self.pipeline([dataset[text_idx]], rep_token=pos, **kwargs)
                text_results.append(result[0])

            # Average the readings for this text
            merged = {}
            for layer in text_results[0].keys():
                merged[layer] = torch.mean(torch.stack(
                    [r[layer] for r in text_results]
                ), dim=0)
            all_results.append(merged)

        return all_results

    def get_directions(self,
                      dataset: List[torch.Tensor],
                      search_sequence: Optional[torch.Tensor] = None,
                      context: Tuple[int, int] = (0, 0),
                      **kwargs) -> Dict:
        """Get representation directions based on token sequence search with context"""
        if search_sequence is None:
            return self.pipeline.get_directions(dataset, **kwargs)

        # Tokenize the search sequence
        search_tokens = self.tokenizer.encode(search_sequence, add_special_tokens=False, return_tensors='pt').to(self.pipeline.device)[0]

        # Get positions for each text
        all_positions = []
        for tokens in dataset:
            seq_pos = self._find_token_sequence(tokens, search_tokens)
            context_pos = self._get_context_slice(tokens, seq_pos, context)
            all_positions.append(context_pos)

        # Get directions for each position
        layer_directions = {}
        for text_idx, positions in enumerate(tqdm(all_positions)):
            for pos in positions:
                dir_result = self.pipeline.get_directions(
                    [dataset[text_idx]],
                    rep_token=pos,
                    **kwargs
                )

                # Initialize or append to layer directions
                for layer, direction in dir_result.items():
                    if layer not in layer_directions:
                        layer_directions[layer] = []
                    layer_directions[layer].append(direction.cpu())

        # Average directions for each layer
        merged_directions = {}
        for layer, directions in layer_directions.items():
            merged_directions[layer] = torch.mean(torch.stack(directions, dim=0), dim=0)

        return merged_directions