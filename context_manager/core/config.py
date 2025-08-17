"""
Configuration management for ContextManager.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7


class MemoryConfig(BaseModel):
    """Configuration for memory management."""
    stm_capacity: int = Field(default=8000, description="Max tokens in short-term memory")
    chunk_size: int = Field(default=2000, description="Tokens per summarization chunk")
    recent_k: int = Field(default=5, description="Last N turns always in context")
    ltm_hits_k: int = Field(default=7, description="Number of LTM hits to retrieve")
    prompt_token_budget: int = Field(default=12000, description="Max tokens for final prompt")
    summary_compression_ratio: float = Field(default=0.3, description="Compression ratio for summaries")
    # Ingestion chunking
    ingestion_chunk_size: int = Field(default=400, description="Tokens per ingestion chunk")
    ingestion_chunk_overlap: int = Field(default=50, description="Token overlap between ingestion chunks")
    # Ranking weights
    similarity_weight: float = Field(default=1.0, description="Weight for vector similarity in scoring")
    recency_weight: float = Field(default=0.0, description="Weight for recency in scoring")
    importance_weight: float = Field(default=0.0, description="Weight for importance (from metadata) in scoring")
    recency_half_life_seconds: float = Field(default=604800.0, description="Half-life (seconds) for recency decay (default 7 days)")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    provider: Literal["faiss"] = "faiss"
    dimension: int = 768  # Default for sentence-transformers
    index_type: str = "HNSW"  # or "Flat"
    metric: str = "cosine"


class Config(BaseModel):
    """Main configuration for ContextManager."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    # Debug and logging
    debug: bool = False
    log_level: str = "INFO"
    
    # Async settings
    async_summarization: bool = True
    background_summarization: bool = True 