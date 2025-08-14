"""
Hierarchical summarization for ContextManager.
"""

from typing import List, Optional, Dict, Any
from loguru import logger
from ..llm.adapters import LLMAdapter, create_llm_adapter
from ..memory.short_term import Turn


class HierarchicalSummarizer:
    """Hierarchical summarization using LLM."""
    
    def __init__(self, llm_adapter: Optional[LLMAdapter] = None, compression_ratio: float = 0.3):
        """
        Initialize hierarchical summarizer.
        
        Args:
            llm_adapter: LLM adapter for summarization
            compression_ratio: Target compression ratio for summaries
        """
        self.llm_adapter = llm_adapter or create_llm_adapter("openai")
        self.compression_ratio = compression_ratio
        
        # Summarization prompts
        self.chunk_summary_prompt = """
Summarize the following conversation chunk in a concise way, preserving key information, decisions, and context that might be important for future reference. Focus on:

1. Main topics discussed
2. Key decisions made
3. Important facts or information shared
4. User preferences or requirements mentioned
5. Any action items or next steps

Conversation chunk:
{text}

Summary:"""

        self.hierarchical_summary_prompt = """
Create a higher-level summary of the following conversation summaries. This should provide an overview of the entire conversation session, highlighting:

1. Overall purpose and goals
2. Major topics covered
3. Key outcomes and decisions
4. Important patterns or themes
5. Context that would be useful for future interactions

Summaries to combine:
{summaries}

Hierarchical Summary:"""
    
    def summarize_chunk(self, turns: List[Turn]) -> str:
        """
        Summarize a chunk of conversation turns.
        
        Args:
            turns: List of conversation turns
            
        Returns:
            Summary text
        """
        if not turns:
            return ""
        
        # Combine turn texts
        combined_text = "\n\n".join([turn.text for turn in turns])
        
        # Create prompt
        prompt = self.chunk_summary_prompt.format(text=combined_text)
        
        try:
            # Generate summary
            summary = self.llm_adapter.generate_sync(
                prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            logger.debug(f"Generated chunk summary for {len(turns)} turns")
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating chunk summary: {e}")
            # Fallback: simple concatenation
            return f"Conversation chunk with {len(turns)} turns: {combined_text[:200]}..."
    
    def summarize_hierarchically(self, summaries: List[str]) -> str:
        """
        Create a hierarchical summary from multiple summaries.
        
        Args:
            summaries: List of summaries to combine
            
        Returns:
            Hierarchical summary
        """
        if not summaries:
            return ""
        
        if len(summaries) == 1:
            return summaries[0]
        
        # Combine summaries
        combined_summaries = "\n\n".join([f"- {summary}" for summary in summaries])
        
        # Create prompt
        prompt = self.hierarchical_summary_prompt.format(summaries=combined_summaries)
        
        try:
            # Generate hierarchical summary
            hierarchical_summary = self.llm_adapter.generate_sync(
                prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            logger.debug(f"Generated hierarchical summary from {len(summaries)} summaries")
            return hierarchical_summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating hierarchical summary: {e}")
            # Fallback: simple concatenation
            return f"Session summary: {len(summaries)} conversation chunks combined."

    def summarize_texts(self, texts: List[str]) -> str:
        """
        Summarize a list of raw texts (not Turn objects).
        """
        if not texts:
            return ""
        combined_text = "\n\n".join(texts)
        prompt = self.chunk_summary_prompt.format(text=combined_text)
        try:
            summary = self.llm_adapter.generate_sync(
                prompt,
                max_tokens=500,
                temperature=0.3
            )
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary from texts: {e}")
            # Fallback: truncate combined text
            return (combined_text[:200] + "...") if len(combined_text) > 200 else combined_text
    
    def estimate_summary_length(self, original_text: str) -> int:
        """
        Estimate the length of a summary based on compression ratio.
        
        Args:
            original_text: Original text
            
        Returns:
            Estimated summary length in characters
        """
        return int(len(original_text) * self.compression_ratio)
    
    def create_summary_metadata(self, turns: List[Turn], summary: str) -> Dict[str, Any]:
        """
        Create metadata for a summary.
        
        Args:
            turns: Original turns that were summarized
            summary: Generated summary
            
        Returns:
            Metadata dictionary
        """
        import time
        
        return {
            "num_turns": len(turns),
            "start_timestamp": turns[0].timestamp if turns else time.time(),
            "end_timestamp": turns[-1].timestamp if turns else time.time(),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / sum(len(turn.text) for turn in turns) if turns else 0,
            "type": "chunk_summary"
        }
    
    def create_hierarchical_metadata(self, summaries: List[str], hierarchical_summary: str) -> Dict[str, Any]:
        """
        Create metadata for a hierarchical summary.
        
        Args:
            summaries: Original summaries
            hierarchical_summary: Generated hierarchical summary
            
        Returns:
            Metadata dictionary
        """
        import time
        
        return {
            "num_summaries": len(summaries),
            "timestamp": time.time(),
            "summary_length": len(hierarchical_summary),
            "type": "hierarchical_summary"
        } 