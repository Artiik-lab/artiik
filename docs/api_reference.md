# API Reference

This section provides complete API documentation for ContextManager, including all classes, methods, configuration options, and usage examples.

## 📚 Table of Contents

- [ContextManager](#contextmanager)
- [Config](#config)
- [MemoryConfig](#memoryconfig)
- [LLMConfig](#llmconfig)
- [VectorStoreConfig](#vectorstoreconfig)
- [ShortTermMemory](#shorttermmemory)
- [LongTermMemory](#longtermmemory)
- [HierarchicalSummarizer](#hierarchicalsummarizer)
- [TokenCounter](#tokencounter)
- [EmbeddingProvider](#embeddingprovider)
- [LLM Adapters](#llm-adapters)

## ContextManager

The main class that orchestrates all memory and context management functionality.

### Constructor

```python
class ContextManager:
    def __init__(
        self,
        config: Optional[Config] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        allow_cross_session: bool = False,
        allow_cross_task: bool = False,
    ):
        """
        Initialize ContextManager.
        
        Args:
            config: Configuration object. If None, uses default configuration.
            session_id: Optional session scope for memory isolation.
            task_id: Optional task scope for memory isolation.
            allow_cross_session: If True, retrieval can include memories from other sessions.
            allow_cross_task: If True, retrieval can include memories from other tasks.
        """
```

### Methods

#### observe()

```python
def observe(self, user_input: str, assistant_response: str) -> None:
    """
    Observe a new conversation turn and update memory.
    
    Args:
        user_input: User's input text
        assistant_response: Assistant's response text
        
    Example:
        cm.observe("Hello!", "Hi there! How can I help you?")
    """
```

#### build_context()

```python
def build_context(self, user_input: str) -> str:
    """
    Build optimized context for LLM call.
    
    Args:
        user_input: Current user input
        
    Returns:
        Optimized context string ready for LLM
        
    Example:
        context = cm.build_context("What did we discuss about Python?")
        response = llm.generate(context)
        
    Notes:
        - Performs similarity search over LTM and applies a lightweight keyword filter to prioritize matching items (hybrid retrieval).
        - Applies budget-aware pruning: removes LTM hits first, then recent turns, before hard truncation to token budget.
    """
```

#### query_memory()

```python
def query_memory(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Query long-term memory for relevant information.
    
    Args:
        query: Search query text
        k: Number of results to return (default: 5)
        
    Returns:
        List of (text, similarity_score) tuples, sorted by similarity
        
    Example:
        results = cm.query_memory("budget planning", k=3)
        for text, score in results:
            print(f"Score {score:.2f}: {text}")
    """
```

#### add_memory()

```python
def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Manually add a memory entry to long-term memory.
    
    Args:
        text: Text to remember
        metadata: Optional metadata dictionary
        
    Returns:
        Memory entry ID
        
    Example:
        memory_id = cm.add_memory(
            "User prefers Python over JavaScript",
            {"type": "preference", "topic": "programming"}
        )
    """
```

#### save_memory()

```python
def save_memory(self, directory_path: str) -> None:
    """
    Persist long-term memory (FAISS index + entries metadata) to disk.
    
    Args:
        directory_path: Target directory to write `index.faiss` and `entries.json`.
    """
```

#### load_memory()

```python
def load_memory(self, directory_path: str) -> None:
    """
    Load long-term memory from disk (expects `index.faiss` and `entries.json`).
    
    Args:
        directory_path: Source directory containing persisted files.
    """
```

#### set_session()

```python
def set_session(self, session_id: Optional[str], allow_cross_session: Optional[bool] = None) -> None:
    """Update session scope and optional cross-session flag."""
```

#### set_task()

```python
def set_task(self, task_id: Optional[str], allow_cross_task: Optional[bool] = None) -> None:
    """Update task scope and optional cross-task flag."""
```

#### get_stats()

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Get statistics about the context manager.
    
    Returns:
        Dictionary containing memory statistics
        
    Example:
        stats = cm.get_stats()
        print(f"STM turns: {stats['short_term_memory']['num_turns']}")
        print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
    """
```

#### clear_memory()

```python
def clear_memory(self) -> None:
    """
    Clear all memory (STM and LTM).
    
    Example:
        cm.clear_memory()
        print("All memory cleared")
    """
```

#### debug_context_building()

```python
def debug_context_building(self, user_input: str) -> Dict[str, Any]:
    """
    Debug context building process.
    
    Args:
        user_input: User input to debug
        
    Returns:
        Dictionary containing debug information
        
    Example:
        debug_info = cm.debug_context_building("What did we discuss?")
        print(f"Recent turns: {debug_info['recent_turns_count']}")
        print(f"LTM hits: {debug_info['ltm_results_count']}")
        print(f"Final tokens: {debug_info['final_context_tokens']}")
    """
```

## Config

Main configuration class for ContextManager.

### Constructor

```python
class Config(BaseModel):
    def __init__(self, **data):
        """
        Initialize configuration with default values.
        
        Args:
            **data: Configuration overrides
        """
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory` | `MemoryConfig` | `MemoryConfig()` | Memory management settings |
| `llm` | `LLMConfig` | `LLMConfig()` | LLM provider settings |
| `vector_store` | `VectorStoreConfig` | `VectorStoreConfig()` | Vector store settings |
| `debug` | `bool` | `False` | Enable debug mode |
| `log_level` | `str` | `"INFO"` | Logging level |
| `async_summarization` | `bool` | `True` | Enable async summarization |
| `background_summarization` | `bool` | `True` | Enable background summarization |

### Example

```python
from context_manager import Config

config = Config(
    memory=MemoryConfig(
        stm_capacity=8000,
        chunk_size=2000,
        recent_k=5,
        ltm_hits_k=7,
        prompt_token_budget=12000,
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key",
    ),
    debug=True,
    log_level="DEBUG"
)
```

## MemoryConfig

Configuration for memory management settings.

### Constructor

```python
class MemoryConfig(BaseModel):
    def __init__(self, **data):
        """
        Initialize memory configuration.
        
        Args:
            **data: Configuration overrides
        """
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `stm_capacity` | `int` | `8000` | Max tokens in short-term memory |
| `chunk_size` | `int` | `2000` | Tokens per summarization chunk |
| `recent_k` | `int` | `5` | Recent turns always in context |
| `ltm_hits_k` | `int` | `7` | Number of LTM results to retrieve |
| `prompt_token_budget` | `int` | `12000` | Max tokens for final context |
| `summary_compression_ratio` | `float` | `0.3` | Target compression ratio |
| `similarity_weight` | `float` | `1.0` | Weight for vector similarity in ranking |
| `recency_weight` | `float` | `0.0` | Weight for recency (newer memories rank higher) |
| `importance_weight` | `float` | `0.0` | Weight for `metadata.importance` in ranking |
| `recency_half_life_seconds` | `float` | `604800.0` | Exponential half-life (seconds) for recency |

### Example

```python
from context_manager import MemoryConfig

memory_config = MemoryConfig(
    stm_capacity=10000,           # Larger STM
    chunk_size=3000,              # Larger chunks
    recent_k=8,                   # More recent turns
    ltm_hits_k=10,               # More LTM results
    prompt_token_budget=16000,    # Larger context budget
    summary_compression_ratio=0.25,  # More aggressive compression
)
```

## LLMConfig

Configuration for LLM provider settings.

### Constructor

```python
class LLMConfig(BaseModel):
    def __init__(self, **data):
        """
        Initialize LLM configuration.
        
        Args:
            **data: Configuration overrides
        """
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `Literal["openai", "anthropic"]` | `"openai"` | LLM provider |
| `model` | `str` | `"gpt-4"` | Model name |
| `api_key` | `Optional[str]` | `None` | API key |
| `max_tokens` | `int` | `1000` | Response token limit |
| `temperature` | `float` | `0.7` | Creativity level (0.0-1.0) |

### Example

```python
from context_manager import LLMConfig

llm_config = LLMConfig(
    provider="anthropic",
    model="claude-3-sonnet-20240229",
    api_key="your-anthropic-key",
    max_tokens=1500,
    temperature=0.5,
)
```

## VectorStoreConfig

Configuration for vector store settings.

### Constructor

```python
class VectorStoreConfig(BaseModel):
    def __init__(self, **data):
        """
        Initialize vector store configuration.
        
        Args:
            **data: Configuration overrides
        """
```

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `Literal["faiss"]` | `"faiss"` | Vector database provider |
| `dimension` | `int` | `768` | Embedding dimension |
| `index_type` | `str` | `"HNSW"` | FAISS index type |
| `metric` | `str` | `"cosine"` | Similarity metric |

### Example

```python
from context_manager import VectorStoreConfig

vector_config = VectorStoreConfig(
    provider="faiss",
    dimension=384,              # Smaller embeddings
    index_type="HNSW",         # Hierarchical Navigable Small World
    metric="cosine",           # Cosine similarity
)
```

## ShortTermMemory

Short-term memory implementation using a deque with token-aware management.

### Constructor

```python
class ShortTermMemory:
    def __init__(self, max_tokens: int = 8000, token_counter: Optional[TokenCounter] = None):
        """
        Initialize short-term memory.
        
        Args:
            max_tokens: Maximum tokens to keep in STM
            token_counter: Token counter instance
        """
```

### Methods

#### add_turn()

```python
def add_turn(self, user_input: str, assistant_response: str) -> None:
    """
    Add a new turn to short-term memory.
    
    Args:
        user_input: User's input
        assistant_response: Assistant's response
    """
```

#### get_recent_turns()

```python
def get_recent_turns(self, k: int) -> List[Turn]:
    """
    Get the k most recent turns.
    
    Args:
        k: Number of recent turns to get
        
    Returns:
        List of recent turns
    """
```

#### get_recent_texts()

```python
def get_recent_texts(self, k: int) -> List[str]:
    """
    Get the text of the k most recent turns.
    
    Args:
        k: Number of recent turns to get
        
    Returns:
        List of turn texts
    """
```

#### get_chunk_for_summarization()

```python
def get_chunk_for_summarization(self, chunk_size: int) -> Tuple[List[Turn], int]:
    """
    Get a chunk of turns for summarization.
    
    Args:
        chunk_size: Target token size for chunk
        
    Returns:
        Tuple of (turns, actual_token_count)
    """
```

#### get_stats()

```python
def get_stats(self) -> dict:
    """
    Get statistics about STM.
    
    Returns:
        Dictionary with STM statistics
    """
```

#### clear()

```python
def clear(self) -> None:
    """
    Clear all turns from STM.
    """
```

### Example

```python
from context_manager.memory.short_term import ShortTermMemory

stm = ShortTermMemory(max_tokens=5000)
stm.add_turn("Hello!", "Hi there!")
stm.add_turn("How are you?", "I'm doing well!")

recent = stm.get_recent_turns(2)
print(f"Recent turns: {len(recent)}")

stats = stm.get_stats()
print(f"Current tokens: {stats['current_tokens']}")
```

## LongTermMemory

Long-term memory implementation using FAISS for vector storage.

### Constructor

```python
class LongTermMemory:
    def __init__(self, dimension: int = 384, embedding_provider: Optional[EmbeddingProvider] = None):
        """
        Initialize long-term memory.
        
        Args:
            dimension: Embedding dimension
            embedding_provider: Embedding provider instance
        """
```

### Methods

#### add_memory()

```python
def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Add a memory entry to long-term memory.
    
    Args:
        text: Text to store
        metadata: Optional metadata
        
    Returns:
        Memory entry ID
    """
```

#### search()

```python
def search(self, query: str, k: int = 7) -> List[Tuple[MemoryEntry, float]]:
    """
    Search for similar memories.
    
    Args:
        query: Query text
        k: Number of results to return
        
    Returns:
        List of (memory_entry, similarity_score) tuples
    """
```

#### search_by_embedding()

```python
def search_by_embedding(self, query_embedding: np.ndarray, k: int = 7) -> List[Tuple[MemoryEntry, float]]:
    """
    Search for similar memories using a pre-computed embedding.
    
    Args:
        query_embedding: Query embedding
        k: Number of results to return
        
    Returns:
        List of (memory_entry, similarity_score) tuples
    """
```

#### get_memory_by_id()

```python
def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
    """
    Get a memory entry by ID.
    
    Args:
        memory_id: Memory entry ID
        
    Returns:
        Memory entry or None if not found
    """
```

#### delete_memory()

```python
def delete_memory(self, memory_id: str) -> bool:
    """
    Delete a memory entry.
    
    Args:
        memory_id: Memory entry ID
        
    Returns:
        True if deleted, False if not found
    """
```

#### get_stats()

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Get statistics about LTM.
    
    Returns:
        Dictionary with LTM statistics
    """
```

#### clear()
#### save()

```python
def save(self, directory_path: str) -> None:
    """
    Save FAISS index and entries metadata to `directory_path`.
    Writes `index.faiss` and `entries.json`.
    """
```

#### load()

```python
def load(self, directory_path: str) -> None:
    """
    Load FAISS index and entries metadata from `directory_path`.
    Expects `index.faiss` and `entries.json`.
    """
```

```python
def clear(self) -> None:
    """
    Clear all memory entries.
    """
```

### Example

```python
from context_manager.memory.long_term import LongTermMemory

ltm = LongTermMemory(dimension=384)

# Add memories
memory_id1 = ltm.add_memory("I like Python programming", {"type": "preference"})
memory_id2 = ltm.add_memory("JavaScript is also useful", {"type": "fact"})

# Search memories
results = ltm.search("programming languages", k=3)
for entry, score in results:
    print(f"Score {score:.2f}: {entry.text}")

# Get statistics
stats = ltm.get_stats()
print(f"Total entries: {stats['num_entries']}")
```

## HierarchicalSummarizer

Hierarchical summarization using LLM.

### Constructor

```python
class HierarchicalSummarizer:
    def __init__(self, llm_adapter: Optional[LLMAdapter] = None, compression_ratio: float = 0.3):
        """
        Initialize hierarchical summarizer.
        
        Args:
            llm_adapter: LLM adapter for summarization
            compression_ratio: Target compression ratio for summaries
        """
```

### Methods

#### summarize_chunk()

```python
def summarize_chunk(self, turns: List[Turn]) -> str:
    """
    Summarize a chunk of conversation turns.
    
    Args:
        turns: List of conversation turns
        
    Returns:
        Summary text
    """
```

#### summarize_hierarchically()
#### summarize_texts()

```python
def summarize_texts(self, texts: List[str]) -> str:
    """
    Summarize a list of raw texts. Falls back to truncated content if LLM errors.
    """
```

```python
def summarize_hierarchically(self, summaries: List[str]) -> str:
    """
    Create a hierarchical summary from multiple summaries.
    
    Args:
        summaries: List of summaries to combine
        
    Returns:
        Hierarchical summary
    """
```

#### estimate_summary_length()

```python
def estimate_summary_length(self, original_text: str) -> int:
    """
    Estimate the length of a summary based on compression ratio.
    
    Args:
        original_text: Original text
        
    Returns:
        Estimated summary length in characters
    """
```

#### create_summary_metadata()

```python
def create_summary_metadata(self, turns: List[Turn], summary: str) -> Dict[str, Any]:
    """
    Create metadata for a summary.
    
    Args:
        turns: Original turns that were summarized
        summary: Generated summary
        
    Returns:
        Metadata dictionary
    """
```

### Example

```python
from context_manager.memory.summarizer import HierarchicalSummarizer
from context_manager.llm.adapters import create_llm_adapter

llm_adapter = create_llm_adapter("openai", api_key="your-key")
summarizer = HierarchicalSummarizer(llm_adapter, compression_ratio=0.3)

# Summarize chunk
turns = [turn1, turn2, turn3]  # Your conversation turns
summary = summarizer.summarize_chunk(turns)
print(f"Chunk summary: {summary}")

# Hierarchical summarization
summaries = ["Summary 1", "Summary 2", "Summary 3"]
hierarchical_summary = summarizer.summarize_hierarchically(summaries)
print(f"Hierarchical summary: {hierarchical_summary}")
```

## TokenCounter

Token counting utilities supporting multiple LLM providers.

### Constructor

```python
class TokenCounter:
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for tokenizer
        """
```

### Methods

#### count_tokens()

```python
def count_tokens(self, text: Union[str, List[str]]) -> int:
    """
    Count tokens in text.
    
    Args:
        text: Text or list of texts to count tokens for
        
    Returns:
        Number of tokens
    """
```

#### count_tokens_dict()

```python
def count_tokens_dict(self, data: Dict[str, Any]) -> int:
    """
    Count tokens in a dictionary structure.
    
    Args:
        data: Dictionary containing text data
        
    Returns:
        Total token count
    """
```

#### truncate_to_tokens()

```python
def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
```

#### estimate_tokens()

```python
def estimate_tokens(self, text: str) -> int:
    """
    Estimate token count without using tokenizer (faster for large texts).
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
```

### Example

```python
from context_manager.utils.token_counter import TokenCounter

counter = TokenCounter("gpt-4")

# Count tokens
text = "Hello, world! This is a test."
token_count = counter.count_tokens(text)
print(f"Tokens: {token_count}")

# Truncate text
long_text = "Very long text..." * 1000
truncated = counter.truncate_to_tokens(long_text, 100)
print(f"Truncated length: {len(truncated)}")

# Estimate tokens
estimate = counter.estimate_tokens(long_text)
print(f"Estimated tokens: {estimate}")
```

## EmbeddingProvider

Provider for text embeddings using sentence-transformers.

### Constructor

```python
class EmbeddingProvider:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding provider.
        
        Args:
            model_name: Sentence transformer model name
        """
```

### Methods

#### embed()

```python
def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
    """
    Generate embeddings for text(s).
    
    Args:
        texts: Text or list of texts to embed
        
    Returns:
        Embeddings as numpy array
    """
```

#### embed_single()

```python
def embed_single(self, text: str) -> np.ndarray:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text to embed
        
    Returns:
        Single embedding as numpy array
    """
```

#### similarity()

```python
def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        
    Returns:
        Cosine similarity score
    """
```

#### batch_similarity()

```python
def batch_similarity(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate similarities between query embedding and batch of embeddings.
    
    Args:
        query_embedding: Query embedding
        embeddings: Batch of embeddings
        
    Returns:
        Array of similarity scores
    """
```

#### get_model_info()

```python
def get_model_info(self) -> Dict[str, any]:
    """
    Get information about the current model.
    
    Returns:
        Dictionary with model information
    """
```

#### list_available_models()

```python
def list_available_models(self) -> Dict[str, Dict[str, any]]:
    """
    List all available models with their specifications.
    
    Returns:
        Dictionary of available models
    """
```

### Example

```python
from context_manager.llm.embeddings import EmbeddingProvider

provider = EmbeddingProvider("all-MiniLM-L6-v2")

# Generate embeddings
texts = ["Hello world", "Goodbye world"]
embeddings = provider.embed(texts)
print(f"Embeddings shape: {embeddings.shape}")

# Single embedding
embedding = provider.embed_single("Hello world")
print(f"Single embedding shape: {embedding.shape}")

# Similarity
similarity = provider.similarity(embedding, embedding)
print(f"Self-similarity: {similarity}")

# Model information
model_info = provider.get_model_info()
print(f"Model size: {model_info['size_mb']}MB")
```

## LLM Adapters

LLM adapters provide a unified interface for different LLM providers.

### OpenAIAdapter

```python
class OpenAIAdapter(LLMAdapter):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key
            model: Model name
        """
```

### AnthropicAdapter

```python
class AnthropicAdapter(LLMAdapter):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic adapter.
        
        Args:
            api_key: Anthropic API key
            model: Model name
        """
```

### Factory Function

```python
def create_llm_adapter(provider: str, **kwargs) -> LLMAdapter:
    """
    Factory function to create LLM adapter.
    
    Args:
        provider: LLM provider ("openai" or "anthropic")
        **kwargs: Additional arguments for the adapter
        
    Returns:
        LLM adapter instance
    """
```

### Example

```python
from context_manager.llm.adapters import create_llm_adapter

# Create OpenAI adapter
openai_adapter = create_llm_adapter("openai", api_key="your-key", model="gpt-4")
response = openai_adapter.generate_sync("Hello, world!")

# Create Anthropic adapter
anthropic_adapter = create_llm_adapter("anthropic", api_key="your-key", model="claude-3-sonnet")
response = anthropic_adapter.generate_sync("Hello, world!")
```

---

**Next**: [Examples](./examples.md) → Practical examples and use cases 