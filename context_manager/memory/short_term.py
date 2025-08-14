"""
Short-term memory implementation for ContextManager.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from loguru import logger
from ..utils.token_counter import TokenCounter


@dataclass
class Turn:
    """Represents a single conversation turn."""
    user_input: str
    assistant_response: str
    token_count: int
    timestamp: float
    
    @property
    def text(self) -> str:
        """Get the full text of this turn."""
        return f"User: {self.user_input}\nAssistant: {self.assistant_response}"
    
    @property
    def user_text(self) -> str:
        """Get just the user input text."""
        return f"User: {self.user_input}"
    
    @property
    def assistant_text(self) -> str:
        """Get just the assistant response text."""
        return f"Assistant: {self.assistant_response}"


class ShortTermMemory:
    """Short-term memory using a deque with token-aware management."""
    
    def __init__(self, max_tokens: int = 8000, token_counter: Optional[TokenCounter] = None):
        """
        Initialize short-term memory.
        
        Args:
            max_tokens: Maximum tokens to keep in STM
            token_counter: Token counter instance
        """
        self.max_tokens = max_tokens
        self.token_counter = token_counter or TokenCounter()
        self.turns: deque[Turn] = deque()
        self.current_tokens = 0
    
    def add_turn(self, user_input: str, assistant_response: str) -> None:
        """
        Add a new turn to short-term memory.
        
        Args:
            user_input: User's input
            assistant_response: Assistant's response
        """
        # Count tokens for this turn
        turn_text = f"User: {user_input}\nAssistant: {assistant_response}"
        token_count = self.token_counter.count_tokens(turn_text)
        
        # Create turn object
        import time
        turn = Turn(
            user_input=user_input,
            assistant_response=assistant_response,
            token_count=token_count,
            timestamp=time.time()
        )
        
        # Add to memory
        self.turns.append(turn)
        self.current_tokens += token_count
        
        # Evict if necessary
        self._evict_if_needed()
        
        logger.debug(f"Added turn to STM. Current tokens: {self.current_tokens}/{self.max_tokens}")
    
    def _evict_if_needed(self) -> None:
        """Evict oldest turns if we exceed max tokens."""
        while self.current_tokens > self.max_tokens and self.turns:
            oldest_turn = self.turns.popleft()
            self.current_tokens -= oldest_turn.token_count
            logger.debug(f"Evicted turn from STM. Remaining tokens: {self.current_tokens}")
    
    def get_recent_turns(self, k: int) -> List[Turn]:
        """
        Get the k most recent turns.
        
        Args:
            k: Number of recent turns to get
            
        Returns:
            List of recent turns
        """
        return list(self.turns)[-k:]
    
    def get_recent_texts(self, k: int) -> List[str]:
        """
        Get the text of the k most recent turns.
        
        Args:
            k: Number of recent turns to get
            
        Returns:
            List of turn texts
        """
        recent_turns = self.get_recent_turns(k)
        return [turn.text for turn in recent_turns]
    
    def get_chunk_for_summarization(self, chunk_size: int) -> Tuple[List[Turn], int]:
        """
        Get a chunk of turns for summarization.
        
        Args:
            chunk_size: Target token size for chunk
            
        Returns:
            Tuple of (turns, actual_token_count)
        """
        chunk = []
        total_tokens = 0
        
        while self.turns and total_tokens < chunk_size:
            turn = self.turns.popleft()
            chunk.append(turn)
            total_tokens += turn.token_count
        
        # Update current token count
        self.current_tokens -= total_tokens
        
        return chunk, total_tokens
    
    def get_all_turns(self) -> List[Turn]:
        """Get all turns in STM."""
        return list(self.turns)
    
    def clear(self) -> None:
        """Clear all turns from STM."""
        self.turns.clear()
        self.current_tokens = 0
        logger.debug("Cleared STM")
    
    def get_stats(self) -> dict:
        """Get statistics about STM."""
        return {
            "num_turns": len(self.turns),
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "utilization": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0
        }
    
    def is_empty(self) -> bool:
        """Check if STM is empty."""
        return len(self.turns) == 0
    
    def __len__(self) -> int:
        """Get number of turns in STM."""
        return len(self.turns) 