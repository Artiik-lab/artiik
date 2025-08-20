# API Reference

Complete API documentation for ContextManager.

## Core Classes

### ContextManager

The main class for managing context and memory.

```python
from artiik import ContextManager
```

#### Constructor

```python
ContextManager(
    config: Optional[Config] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    allow_cross_session: bool = False,
    allow_cross_task: bool = False,
)
```

**Parameters:**
- `config`: Configuration object (optional)
- `session_id`: Session identifier for memory isolation (optional)
- `task_id`: Task identifier for memory isolation (optional)
- `allow_cross_session`: Allow cross-session memory access (default: False)
- `allow_cross_task`: Allow cross-task memory access (default: False)

#### Methods

##### observe(user_input: str, assistant_response: str) -> None

Store a conversation turn in memory.

```python
cm.observe("Hello!", "Hi there!")
```

##### build_context(user_input: str) -> str

Build optimized context for LLM input.

```python
context = cm.build_context("What did we discuss?")
```

##### query_memory(query: str, k: int = 5) -> List[Tuple[str, float]]

Query memory for relevant information.

```python
results = cm.query_memory("Python programming", k=10)
for text, score in results:
    print(f"Score {score:.2f}: {text}")
```

##### ingest_file(file_path: str, importance: float = 0.5) -> int

Ingest a file into long-term memory.

```python
chunks = cm.ingest_file("docs/README.md", importance=0.8)
print(f"Ingested {chunks} chunks")
```

##### ingest_directory(directory_path: str, file_types: List[str] = None, recursive: bool = True, importance: float = 0.5) -> int

Ingest a directory of files into long-term memory.

```python
total = cm.ingest_directory(
    "./my_repo",
    file_types=[".py", ".md"],
    recursive=True,
    importance=0.7
)
print(f"Total chunks ingested: {total}")
```

##### ingest_text(text: str, importance: float = 0.5) -> int

Ingest raw text into long-term memory.

```python
chunks = cm.ingest_text("This is some important information.", importance=0.9)
```

##### get_stats() -> Dict[str, Any]

Get memory statistics.

```python
stats = cm.get_stats()
print(f"STM turns: {stats['short_term_memory']['num_turns']}")
print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
```

##### debug_context_building(user_input: str) -> Dict[str, Any]

Get debug information about context building.

```python
debug_info = cm.debug_context_building("What did we discuss?")
print(f"Recent turns: {debug_info['recent_turns_count']}")
print(f"LTM hits: {debug_info['ltm_results_count']}")
```

##### save_memory(directory_path: str) -> None

Save memory to disk.

```python
cm.save_memory("./memory_backup")
```

##### load_memory(directory_path: str) -> None

Load memory from disk.

```python
cm.load_memory("./memory_backup")
```

### Config

Configuration class for ContextManager.

```python
from artiik import Config, MemoryConfig, LLMConfig, VectorStoreConfig
```

#### Constructor

```python
Config(
    llm: LLMConfig = Field(default_factory=LLMConfig),
    memory: MemoryConfig = Field(default_factory=MemoryConfig),
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig),
    debug: bool = False,
    log_level: str = "INFO",
    async_summarization: bool = True,
    background_summarization: bool = True,
)
```

### MemoryConfig

Configuration for memory management.

```python
MemoryConfig(
    stm_capacity: int = 8000,              # Max tokens in short-term memory
    chunk_size: int = 2000,                # Tokens per summarization chunk
    recent_k: int = 5,                     # Recent turns always in context
    ltm_hits_k: int = 7,                   # Number of LTM results to retrieve
    prompt_token_budget: int = 12000,      # Max tokens for final context
    summary_compression_ratio: float = 0.3,  # Summary compression target
    ingestion_chunk_size: int = 400,       # Tokens per ingestion chunk
    ingestion_chunk_overlap: int = 50,     # Token overlap between chunks
    similarity_weight: float = 1.0,        # Weight for vector similarity
    recency_weight: float = 0.0,           # Weight for recency
    importance_weight: float = 0.0,        # Weight for importance
    recency_half_life_seconds: float = 604800.0,  # Half-life for recency decay
)
```

### LLMConfig

Configuration for LLM providers.

```python
LLMConfig(
    provider: Literal["openai", "anthropic"] = "openai",
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
)
```

### VectorStoreConfig

Configuration for vector store.

```python
VectorStoreConfig(
    provider: Literal["faiss"] = "faiss",
    dimension: int = 768,
    index_type: str = "HNSW",
    metric: str = "cosine",
)
```

## Memory Components

### ShortTermMemory

Manages recent conversation turns.

```python
from context_manager.memory.short_term import ShortTermMemory
```

#### Methods

- `add_turn(user_input: str, assistant_response: str) -> None`
- `get_recent_turns(k: int) -> List[Turn]`
- `clear() -> None`

### LongTermMemory

Manages long-term semantic memory using FAISS.

```python
from context_manager.memory.long_term import LongTermMemory
```

#### Methods

- `add_memory(text: str, metadata: Dict[str, Any] = None) -> None`
- `search(query: str, k: int = 5) -> List[Tuple[str, float]]`
- `clear() -> None`

## LLM Components

### LLMAdapter

Base class for LLM adapters.

```python
from context_manager.llm.adapters import LLMAdapter, create_llm_adapter
```

#### create_llm_adapter

```python
adapter = create_llm_adapter(
    provider: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4"
)
```

## Embedding Components

### EmbeddingProvider

Manages text embeddings.

```python
from context_manager.llm.embeddings import EmbeddingProvider
```

#### Methods

- `get_embedding(text: str) -> List[float]`
- `get_dimension() -> int`

## Utility Components

### TokenCounter

Counts tokens in text.

```python
from context_manager.utils.token_counter import TokenCounter
```

#### Methods

- `count_tokens(text: str) -> int`
- `truncate_to_tokens(text: str, max_tokens: int) -> str`

## Error Handling

### Common Exceptions

```python
from context_manager.core.exceptions import (
    ContextManagerError,
    ConfigurationError,
    MemoryError,
    LLMError,
)
```

## Configuration Examples

### Basic Configuration

```python
from artiik import Config, ContextManager

config = Config()
cm = ContextManager(config)
```

### Custom Memory Configuration

```python
from artiik import Config, MemoryConfig, ContextManager

config = Config(
    memory=MemoryConfig(
        stm_capacity=12000,
        chunk_size=3000,
        recent_k=8,
        ltm_hits_k=10,
        prompt_token_budget=16000,
        summary_compression_ratio=0.2,
    )
)
cm = ContextManager(config)
```

### Custom LLM Configuration

```python
from artiik import Config, LLMConfig, ContextManager

config = Config(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key",
        max_tokens=1000,
        temperature=0.3,
    )
)
cm = ContextManager(config)
```

### Debug Configuration

```python
from artiik import Config, ContextManager

config = Config(
    debug=True,
    log_level="DEBUG",
)
cm = ContextManager(config)
```

## Performance Considerations

### Memory Usage

- **STM**: Proportional to `stm_capacity`
- **LTM**: Proportional to number of entries × embedding dimension
- **Embedding Model**: ~90MB for default model

### Speed Optimization

```python
config = Config(
    memory=MemoryConfig(
        recent_k=3,        # Fewer recent turns
        ltm_hits_k=5,      # Fewer LTM results
        chunk_size=1000,   # Smaller chunks
    ),
    llm=LLMConfig(
        model="gpt-3.5-turbo",  # Faster model
        max_tokens=500,         # Shorter responses
    )
)
```

### Quality Optimization

```python
config = Config(
    memory=MemoryConfig(
        recent_k=10,       # More recent turns
        ltm_hits_k=10,     # More LTM results
        chunk_size=2000,   # Larger chunks
        prompt_token_budget=16000,  # Larger context budget
    ),
    llm=LLMConfig(
        model="gpt-4",     # Higher quality model
        temperature=0.3,   # More focused responses
    )
)
```

---

**Need examples?** → [Examples](./examples.md) | **Troubleshooting?** → [Troubleshooting](./troubleshooting.md) 