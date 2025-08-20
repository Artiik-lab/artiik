# Architecture

This document describes the system architecture and design decisions behind ContextManager.

## 🏗️ System Overview

ContextManager is built around a modular, component-based architecture that separates concerns and enables easy extension and customization.

```
┌─────────────────────────────────────────────────────────────┐
│                    ContextManager                          │
├─────────────────────────────────────────────────────────────┤
│  Core Orchestrator    │  Memory System     │  LLM System   │
│  ┌─────────────────┐  │  ┌──────────────┐  │  ┌──────────┐  │
│  │ ContextManager  │  │  │ STM + LTM    │  │  │ Adapters │  │
│  │ - Coordination  │  │  │ - FAISS      │  │  │ - OpenAI │  │
│  │ - Configuration │  │  │ - Summarizer │  │  │ - Claude │  │
│  └─────────────────┘  │  └──────────────┘  │  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🧩 Component Architecture

### Core Components

#### ContextManager (Orchestrator)

The main orchestrator that coordinates all components:

```python
class ContextManager:
    def __init__(self, config: Config):
        self.config = config
        self.short_term_memory = ShortTermMemory(config.memory)
        self.long_term_memory = LongTermMemory(config.vector_store)
        self.summarizer = HierarchicalSummarizer(config.llm)
        self.token_counter = TokenCounter(config.llm.model)
```

**Responsibilities**:
- Coordinate memory operations
- Manage context building process
- Handle configuration
- Provide unified API

#### Memory System

Dual-memory architecture with automatic management:

```python
# Short-Term Memory (STM)
class ShortTermMemory:
    def __init__(self, max_tokens: int):
        self.turns: deque[Turn] = deque()
        self.current_tokens = 0
        self.max_tokens = max_tokens

# Long-Term Memory (LTM)
class LongTermMemory:
    def __init__(self, dimension: int):
        self.entries: List[MemoryEntry] = []
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.embedding_provider = EmbeddingProvider()
```

**Responsibilities**:
- STM: Fast access to recent context
- LTM: Semantic storage and retrieval
- Automatic summarization and offloading

#### LLM System

Extensible adapter system for different providers:

```python
class LLMAdapter(ABC):
    @abstractmethod
    def generate_sync(self, prompt: str) -> str:
        pass

class OpenAIAdapter(LLMAdapter):
    def generate_sync(self, prompt: str) -> str:
        # OpenAI API implementation
        pass

class AnthropicAdapter(LLMAdapter):
    def generate_sync(self, prompt: str) -> str:
        # Anthropic API implementation
        pass
```

**Responsibilities**:
- Unified interface for different LLM providers
- Handle API-specific details
- Provide consistent error handling

## 🔄 Data Flow

### Context Building Flow

```
User Input
    ↓
1. Retrieve Recent (STM)
    ↓
2. Search LTM (Vector Similarity)
    ↓
3. Assemble Context
    ↓
4. Optimize Tokens
    ↓
Final Context
```

### Memory Management Flow

```
Conversation Turn
    ↓
Add to STM
    ↓
Check Capacity
    ↓
If Overflow: Summarize & Offload
    ↓
Store in LTM
```

### Summarization Flow

```
Old STM Turns
    ↓
Chunk by Token Size
    ↓
LLM Summarization
    ↓
Store Summary in LTM
    ↓
Clear Old Turns from STM
```

## 📊 Data Structures

### Turn

```python
@dataclass
class Turn:
    user_input: str
    assistant_response: str
    token_count: int
    timestamp: float
    
    @property
    def text(self) -> str:
        return f"User: {self.user_input}\nAssistant: {self.assistant_response}"
```

### MemoryEntry

```python
@dataclass
class MemoryEntry:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
```

### Configuration

```python
class Config(BaseModel):
    llm: LLMConfig
    memory: MemoryConfig
    vector_store: VectorStoreConfig
    debug: bool = False
    log_level: str = "INFO"
```

## 🔧 Configuration System

### Hierarchical Configuration

```python
# Main configuration
config = Config(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-key"
    ),
    memory=MemoryConfig(
        stm_capacity=8000,
        chunk_size=2000,
        recent_k=5,
        ltm_hits_k=7
    ),
    vector_store=VectorStoreConfig(
        provider="faiss",
        dimension=384,
        index_type="HNSW"
    )
)
```

### Configuration Philosophy

1. **Sensible Defaults**: Works out of the box
2. **Explicit Configuration**: All settings configurable
3. **Validation**: Pydantic ensures valid configuration
4. **Composition**: Components can be configured independently

## 🎯 Design Patterns

### Strategy Pattern

Different LLM providers implement the same interface:

```python
def create_llm_adapter(provider: str, **kwargs) -> LLMAdapter:
    if provider == "openai":
        return OpenAIAdapter(**kwargs)
    elif provider == "anthropic":
        return AnthropicAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### Observer Pattern

Memory components automatically react to state changes:

```python
class ShortTermMemory:
    def add_turn(self, user_input: str, assistant_response: str):
        # Add turn
        self.turns.append(turn)
        self.current_tokens += turn.token_count
        
        # Notify if capacity exceeded
        if self.current_tokens > self.max_tokens:
            self._trigger_summarization()
```

### Factory Pattern

Component creation is abstracted through factories:

```python
class ComponentFactory:
    @staticmethod
    def create_memory_system(config: MemoryConfig) -> Tuple[ShortTermMemory, LongTermMemory]:
        stm = ShortTermMemory(config.stm_capacity)
        ltm = LongTermMemory(config.vector_store.dimension)
        return stm, ltm
```

## 🔒 Error Handling

### Graceful Degradation

```python
def build_context(self, user_input: str) -> str:
    try:
        # Normal context building
        return self._build_context_internal(user_input)
    except Exception as e:
        logger.error(f"Context building failed: {e}")
        # Fallback to basic context
        return self._build_fallback_context(user_input)
```

### Error Recovery

```python
def _summarize_and_offload(self):
    try:
        summary = self.summarizer.summarize_chunk(chunk)
        self.long_term_memory.add_memory(summary)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        # Store raw text as fallback
        self.long_term_memory.add_memory(" ".join([turn.text for turn in chunk]))
```

## ⚡ Performance Considerations

### Memory Optimization

1. **Token-Aware Management**: Exact token counting prevents overflow
2. **Lazy Loading**: Embeddings generated on-demand
3. **Efficient Data Structures**: Deque for STM, FAISS for LTM
4. **Configurable Limits**: Memory usage bounded by configuration

### Speed Optimization

1. **Vector Search**: FAISS provides sub-linear search complexity
2. **Caching**: Embeddings cached after generation
3. **Batch Operations**: Multiple operations batched where possible
4. **Async Support**: Non-blocking operations for I/O

### Scalability

1. **Horizontal Scaling**: Multiple ContextManager instances
2. **Memory Persistence**: Save/load for long-term storage
3. **Configurable Limits**: Adjust for resource constraints
4. **Modular Design**: Components can be optimized independently

## 🔐 Security Considerations

### API Key Management

```python
# Environment variables for sensitive data
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable required")
```

### Data Privacy

1. **Local Processing**: Embeddings generated locally
2. **Configurable Persistence**: User controls data storage
3. **Session Isolation**: Memory scoped by session/task
4. **No External Dependencies**: Core functionality works offline

## 🧪 Testing Architecture

### Component Testing

```python
class TestShortTermMemory:
    def test_add_turn(self):
        stm = ShortTermMemory(max_tokens=1000)
        stm.add_turn("Hello!", "Hi there!")
        assert len(stm.turns) == 1
        assert stm.current_tokens > 0
```

### Integration Testing

```python
class TestContextManager:
    def test_end_to_end_workflow(self):
        cm = ContextManager()
        cm.observe("Hello!", "Hi there!")
        context = cm.build_context("What did we discuss?")
        assert "Hello" in context
```

### Mock Testing

```python
@patch('context_manager.llm.adapters.create_llm_adapter')
def test_with_mock_llm(self, mock_adapter):
    mock_adapter.return_value = MockLLMAdapter()
    cm = ContextManager()
    # Test with mocked dependencies
```

## 🔄 Extension Points

### Custom Memory Implementations

```python
class CustomShortTermMemory(ShortTermMemory):
    def add_turn(self, user_input: str, assistant_response: str):
        # Custom logic
        super().add_turn(user_input, assistant_response)
```

### Custom LLM Adapters

```python
class CustomLLMAdapter(LLMAdapter):
    def generate_sync(self, prompt: str) -> str:
        # Custom LLM implementation
        return "Custom response"
```

### Custom Summarizers

```python
class CustomSummarizer(HierarchicalSummarizer):
    def summarize_chunk(self, turns: List[Turn]) -> str:
        # Custom summarization logic
        return "Custom summary"
```

## 📈 Monitoring and Observability

### Metrics Collection

```python
class MetricsCollector:
    def __init__(self):
        self.context_build_times = []
        self.memory_usage = []
        self.error_counts = {}
    
    def record_context_build_time(self, duration: float):
        self.context_build_times.append(duration)
    
    def get_average_build_time(self) -> float:
        return sum(self.context_build_times) / len(self.context_build_times)
```

### Health Checks

```python
def health_check(self) -> Dict[str, Any]:
    return {
        "status": "healthy",
        "memory_usage": self.get_memory_stats(),
        "performance": self.get_performance_stats(),
        "errors": self.get_error_stats()
    }
```

## 🚀 Deployment Considerations

### Resource Requirements

1. **Memory**: ~90MB for embedding model + configurable STM/LTM
2. **CPU**: Minimal for basic operations, more for summarization
3. **Storage**: Configurable based on memory persistence needs
4. **Network**: Required for LLM API calls and model downloads

### Environment Setup

```bash
# Production environment
export OPENAI_API_KEY="your-key"
export CONTEXT_MANAGER_LOG_LEVEL="INFO"
export CONTEXT_MANAGER_DEBUG="false"

# Development environment
export CONTEXT_MANAGER_LOG_LEVEL="DEBUG"
export CONTEXT_MANAGER_DEBUG="true"
```

### Containerization

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "your_app.py"]
```

---

**Ready to dive deeper?** → [API Reference](./api_reference.md) | **See examples?** → [Examples](./examples.md) 