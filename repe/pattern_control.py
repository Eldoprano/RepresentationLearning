from typing import Dict, Optional, List, Tuple, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import re
from tqdm import tqdm

class PatternRepControl:
    """Controls application of representation patterns with fine-grained control over sections and scope"""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        control_section: str = "full", # "reasoning", "answer" or "full" 
        control_scope: str = "full",   # "full", "first_n", "last"
        n_sentences: int = 1,
        initial_text: Optional[str] = None,
        reasoning_start: str = "<think>",
        reasoning_end: str = "</think>",
        generation_end: str = "<｜end▁of▁sentence｜>",
        block_name: str = "decoder_block"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.control_section = control_section
        self.control_scope = control_scope  
        self.n_sentences = n_sentences
        self.initial_text = initial_text
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.generation_end = generation_end
        self.block_name = block_name

        # Pre-compile regex patterns for faster matching
        self.sentence_split_pattern = re.compile(r'[.!?]')
        self.reasoning_start_pattern = re.compile(re.escape(reasoning_start))
        self.reasoning_end_pattern = re.compile(re.escape(reasoning_end))

        # Cache for tokenized inputs
        self._cached_input_ids = None
        self._cached_prompt = None

    def _get_input_ids(self, text: str) -> torch.Tensor:
        """Get cached or new input IDs"""
        if text == self._cached_prompt and self._cached_input_ids is not None:
            return self._cached_input_ids
            
        self._cached_prompt = text
        self._cached_input_ids = self.tokenizer(
            text, 
            return_tensors="pt"
        ).input_ids.to(self.model.device)
        return self._cached_input_ids

    def _is_in_section(self, text: str, current_token: str) -> Tuple[bool, str]:
        """Optimized section checking"""
        if self.control_section == "full":
            return True, "full"
            
        if self.control_section == "reasoning":
            in_reasoning = bool(self.reasoning_start_pattern.search(text)) and \
                         not bool(self.reasoning_end_pattern.search(text[-20:]))
            return (in_reasoning, "reasoning") if in_reasoning else (False, "")
            
        if self.control_section == "answer":
            in_answer = bool(self.reasoning_end_pattern.search(text)) and \
                       self.generation_end not in text[-20:]
            return (in_answer, "answer") if in_answer else (False, "")
            
        return False, ""

    def _should_apply_control(self, text: str, current_token: str, section: str) -> bool:
        """Optimized control scope checking"""
        if self.control_scope == "full":
            return True
            
        if self.control_scope == "first_n":
            if section == "reasoning":
                relevant_text = text[text.find(self.reasoning_start):]
            elif section == "answer":
                relevant_text = text[text.find(self.reasoning_end)+len(self.reasoning_end):]
            else:
                relevant_text = text
                
            sentences = len([s for s in self.sentence_split_pattern.split(relevant_text) if s.strip()])
            return sentences < self.n_sentences
            
        if self.control_scope == "last":
            if section == "reasoning":
                return "</th" in current_token
            if section == "answer": 
                return "<｜end" in current_token
            return False
            
        return False

    def generate(
        self,
        prompt: str,
        activations: Dict[int, torch.Tensor],
        activation_scale: float = 1.0,
        max_new_tokens: int = 100,
        batch_size: int = 1,
        **kwargs
    ) -> str:
        """Generate text with controlled activations and progress bar"""
        input_ids = self._get_input_ids(prompt)
        
        def modify_activations(module, input, output):
            for layer, direction in activations.items():
                output[layer] = output[layer] + activation_scale * direction.to(output[layer].device)
            return output
            
        hooks = []
        for name, module in self.model.named_modules():
            if self.block_name in name:
                hook = module.register_forward_hook(modify_activations)
                hooks.append(hook)
                
        try:
            generated_text = prompt
            current_token = ""
            output_ids = []
            
            # Add progress bar
            pbar = tqdm(total=max_new_tokens, desc="Generating")
            
            while len(output_ids) < max_new_tokens:
                # Generate next token(s)
                inputs = self._get_input_ids(generated_text)
                
                with torch.no_grad():
                    next_tokens = self.model.generate(
                        **{"input_ids": inputs},
                        max_new_tokens=batch_size,
                        **kwargs
                    )[0, -batch_size:]
                    
                # Process generated tokens
                for token in next_tokens:
                    current_token = self.tokenizer.decode(token)
                    
                    in_section, section = self._is_in_section(generated_text, current_token)
                    apply_control = in_section and self._should_apply_control(
                        generated_text, current_token, section
                    )
                    
                    if apply_control:
                        if self.initial_text and not output_ids:
                            generated_text += self.initial_text
                            current_token = self.initial_text[-1]
                            
                        output_ids.append(token)
                        generated_text += current_token
                        pbar.update(1)
                        
                        if self.generation_end in current_token:
                            break
                            
                    if len(output_ids) >= max_new_tokens:
                        break
                        
            pbar.close()
            return generated_text
            
        finally:
            for hook in hooks:
                hook.remove()
