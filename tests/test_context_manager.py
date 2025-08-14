"""
Tests for ContextManager core functionality.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from context_manager.core import ContextManager, Config
from context_manager.memory.short_term import ShortTermMemory, Turn
from context_manager.memory.long_term import LongTermMemory
from context_manager.utils.token_counter import TokenCounter


class TestShortTermMemory:
    """Test short-term memory functionality."""
    
    def test_add_turn(self):
        """Test adding a turn to STM."""
        stm = ShortTermMemory(max_tokens=1000)
        stm.add_turn("Hello", "Hi there!")
        
        assert len(stm.turns) == 1
        assert stm.turns[0].user_input == "Hello"
        assert stm.turns[0].assistant_response == "Hi there!"
    
    def test_eviction(self):
        """Test that STM evicts when over capacity."""
        stm = ShortTermMemory(max_tokens=100)  # Small capacity
        
        # Add turns until we exceed capacity
        for i in range(10):
            stm.add_turn(f"User message {i}", f"Assistant response {i}")
        
        # Should have evicted some turns
        assert stm.current_tokens <= stm.max_tokens
    
    def test_get_recent_turns(self):
        """Test getting recent turns."""
        stm = ShortTermMemory()
        
        for i in range(5):
            stm.add_turn(f"User {i}", f"Assistant {i}")
        
        recent = stm.get_recent_turns(3)
        assert len(recent) == 3
        assert recent[-1].user_input == "User 4"


class TestLongTermMemory:
    """Test long-term memory functionality."""
    
    def test_add_memory(self):
        """Test adding memory to LTM."""
        ltm = LongTermMemory(dimension=384)
        memory_id = ltm.add_memory("Test memory", {"type": "test"})
        
        assert memory_id.startswith("mem_")
        assert len(ltm.entries) == 1
        assert ltm.entries[0].text == "Test memory"
    
    def test_search(self):
        """Test searching LTM."""
        ltm = LongTermMemory(dimension=384)
        
        # Add some memories
        ltm.add_memory("I like pizza", {"type": "preference"})
        ltm.add_memory("The weather is sunny", {"type": "weather"})
        ltm.add_memory("Python is a programming language", {"type": "fact"})
        
        # Search for pizza
        results = ltm.search("pizza", k=2)
        assert len(results) > 0
        assert any("pizza" in entry.text.lower() for entry, _ in results)


class TestContextManager:
    """Test ContextManager functionality."""
    
    @patch('context_manager.llm.adapters.create_llm_adapter')
    @patch('context_manager.llm.embeddings.EmbeddingProvider')
    def test_initialization(self, mock_embedding, mock_llm):
        """Test ContextManager initialization."""
        # Mock the dependencies
        mock_llm.return_value = Mock()
        mock_embedding.return_value = Mock()
        
        cm = ContextManager()
        
        assert cm.short_term_memory is not None
        assert cm.long_term_memory is not None
        assert cm.summarizer is not None
    
    def test_observe(self):
        """Test observing a conversation turn."""
        with patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
            mock_llm.return_value = Mock()
            
            cm = ContextManager()
            cm.observe("Hello", "Hi there!")
            
            assert len(cm.short_term_memory.turns) == 1
            assert cm.short_term_memory.turns[0].user_input == "Hello"
    
    def test_build_context(self):
        """Test building context."""
        with patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
            mock_llm.return_value = Mock()
            
            cm = ContextManager()
            
            # Add some conversation history
            cm.observe("Hello", "Hi there!")
            cm.observe("How are you?", "I'm doing well!")
            
            # Build context
            context = cm.build_context("What did we talk about?")
            
            assert "Hello" in context
            assert "How are you?" in context
            assert "What did we talk about?" in context
    
    def test_query_memory(self):
        """Test querying memory."""
        with patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
            mock_llm.return_value = Mock()
            
            cm = ContextManager()
            
            # Add some memories
            cm.add_memory("I like pizza", {"type": "preference"})
            cm.add_memory("Python is great", {"type": "opinion"})
            
            # Query memory
            results = cm.query_memory("pizza")
            
            assert len(results) > 0
            assert any("pizza" in text.lower() for text, _ in results)


class TestTokenCounter:
    """Test token counting functionality."""
    
    def test_count_tokens(self):
        """Test token counting."""
        counter = TokenCounter()
        
        # Test basic counting
        count = counter.count_tokens("Hello world")
        assert count > 0
        
        # Test list counting
        count = counter.count_tokens(["Hello", "world"])
        assert count > 0
    
    def test_truncate_to_tokens(self):
        """Test token truncation."""
        counter = TokenCounter()
        
        long_text = "This is a very long text that should be truncated " * 100
        truncated = counter.truncate_to_tokens(long_text, 10)
        
        assert counter.count_tokens(truncated) <= 10


if __name__ == "__main__":
    pytest.main([__file__]) 