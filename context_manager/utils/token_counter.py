"""
Token counting utilities for ContextManager.
"""

import tiktoken
from typing import List, Union, Dict, Any
from loguru import logger


class TokenCounter:
    """Token counter supporting multiple LLM providers."""
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for tokenizer (e.g., "gpt-4", "claude-3-sonnet")
        """
        self.model = model
        self._encoder = None
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize the appropriate tokenizer."""
        try:
            if self.model.startswith("gpt-") or self.model.startswith("text-"):
                # OpenAI models
                self._encoder = tiktoken.encoding_for_model(self.model)
            elif self.model.startswith("claude-"):
                # Anthropic models - use GPT-4 tokenizer as approximation
                self._encoder = tiktoken.encoding_for_model("gpt-4")
            else:
                # Default to GPT-4
                self._encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer for {self.model}: {e}")
            # Fallback to GPT-4
            self._encoder = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: Union[str, List[str]]) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text or list of texts to count tokens for
            
        Returns:
            Number of tokens
        """
        if isinstance(text, list):
            return sum(self.count_tokens(t) for t in text)
        
        if not text:
            return 0
        
        try:
            return len(self._encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_tokens_dict(self, data: Dict[str, Any]) -> int:
        """
        Count tokens in a dictionary structure.
        
        Args:
            data: Dictionary containing text data
            
        Returns:
            Total token count
        """
        total = 0
        for key, value in data.items():
            if isinstance(value, str):
                total += self.count_tokens(value)
            elif isinstance(value, list):
                total += self.count_tokens(value)
            elif isinstance(value, dict):
                total += self.count_tokens_dict(value)
        return total
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            Truncated text
        """
        if not text:
            return text
        
        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode
        truncated_tokens = tokens[:max_tokens]
        return self._encoder.decode(truncated_tokens)

    def split_text_into_token_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> list:
        """
        Split text into token-bounded chunks with optional overlap.

        Args:
            text: Input text to split
            chunk_size: Target tokens per chunk
            overlap: Overlap tokens between consecutive chunks

        Returns:
            List of string chunks
        """
        if not text:
            return []
        if chunk_size <= 0:
            return [text]
        tokens = self._encoder.encode(text)
        chunks = []
        start = 0
        step = max(1, chunk_size - max(0, overlap))
        while start < len(tokens):
            end = min(len(tokens), start + chunk_size)
            chunk = self._encoder.decode(tokens[start:end])
            chunks.append(chunk)
            if end == len(tokens):
                break
            start += step
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count without using tokenizer (faster for large texts).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4 