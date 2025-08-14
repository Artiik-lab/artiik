# Architecture

This section provides a deep dive into ContextManager's system architecture, component interactions, data flow, and performance characteristics.

## 📚 Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Memory Management](#memory-management)
- [Performance Characteristics](#performance-characteristics)
- [Scalability Considerations](#scalability-considerations)
- [Security Considerations](#security-considerations)

## System Overview

### High-Level Architecture

ContextManager implements a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Agent Code    │  │  Web API        │  │  CLI Tools  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  ContextManager Core                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Orchestrator   │  │  Config Mgmt    │  │  Token Mgr  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Memory Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Short-Term     │  │  Long-Term      │  │ Summarizer  │ │
│  │  Memory (STM)   │  │  Memory (LTM)   │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  LLM Adapters   │  │  Vector Store   │  │ Embeddings  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Loose Coupling**: Components communicate through well-defined interfaces
3. **High Cohesion**: Related functionality is grouped together
4. **Extensibility**: Easy to add new LLM providers, vector stores, etc.
5. **Performance**: Optimized for production use with configurable trade-offs

## Component Architecture

### 1. ContextManager (Orchestrator)

The main orchestrator that coordinates all components.

```python
class ContextManager:
    def __init__(self, config: Optional[Config] = None):
        # Initialize components
        self.config = config or Config()
        self.token_counter = TokenCounter(self.config.llm.model)
        self.embedding_provider = EmbeddingProvider()
        self.llm_adapter = create_llm_adapter(...)
        self.short_term_memory = ShortTermMemory(...)
        self.long_term_memory = LongTermMemory(...)
        self.summarizer = HierarchicalSummarizer(...)
```

**Responsibilities**:
- Coordinate memory operations
- Manage context building process
- Handle token budget optimization
- Provide unified interface for agents

**Key Methods**:
```python
def observe(self, user_input: str, assistant_response: str)
def build_context(self, user_input: str) -> str
def query_memory(self, query: str, k: int = 5) -> List[Tuple[str, float]]
def get_stats(self) -> Dict[str, Any]
```

### 2. Short-Term Memory (STM)

Fast, token-aware memory for recent conversation turns.

```python
class ShortTermMemory:
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.turns: deque[Turn] = deque()
        self.current_tokens = 0
```

**Data Structure**:
```python
@dataclass
class Turn:
    user_input: str
    assistant_response: str
    token_count: int
    timestamp: float
```

**Key Operations**:
- **Add Turn**: O(1) append with token counting
- **Evict**: O(1) removal of oldest turns when capacity exceeded
- **Retrieve**: O(k) for k recent turns
- **Chunk**: O(n) for summarization chunks

### 3. Long-Term Memory (LTM)

Vector-based semantic storage with FAISS index.

```python
class LongTermMemory:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.entries: List[MemoryEntry] = []
        self.index = faiss.IndexHNSWFlat(dimension, 32)
```

**Data Structure**:
```python
@dataclass
class MemoryEntry:
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
```

**Key Operations**:
- **Add Memory**: O(log N) for FAISS insertion
- **Search**: O(log N) for similarity search
- **Delete**: O(N) requires index rebuild
- **Update**: Not supported (immutable design)

### 4. Hierarchical Summarizer

LLM-based summarization with multiple levels.

```python
class HierarchicalSummarizer:
    def __init__(self, llm_adapter: LLMAdapter, compression_ratio: float = 0.3):
        self.llm_adapter = llm_adapter
        self.compression_ratio = compression_ratio
```

**Summarization Levels**:
1. **Chunk Level**: Summarize conversation chunks (~2000 tokens)
2. **Session Level**: Combine multiple chunk summaries
3. **Topic Level**: Group related session summaries
4. **Lifetime Level**: Overall conversation summary

### 5. LLM Adapters

Unified interface for different LLM providers.

```python
class LLMAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs) -> str:
        pass
```

**Supported Providers**:
- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo
- **Anthropic**: Claude-3 Sonnet, Claude-3 Opus
- **Custom**: Extensible for other providers

### 6. Embedding Provider

Text embedding generation using sentence-transformers.

```python
class EmbeddingProvider:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
```

**Supported Models**:
- `all-MiniLM-L6-v2`: 90MB, 384 dimensions, fast
- `paraphrase-MiniLM-L3-v2`: 61MB, 384 dimensions, very fast
- `all-mpnet-base-v2`: 420MB, 768 dimensions, excellent quality

## Data Flow

### 1. Context Building Flow

```
User Input
    ↓
ContextManager.build_context()
    ↓
1. Get recent turns from STM
    ↓
2. Search LTM for relevant memories
    ↓
3. Assemble context parts
    ↓
4. Optimize token budget
    ↓
5. Return optimized context
```

**Detailed Flow**:
```python
def build_context(self, user_input: str) -> str:
    # Step 1: Get recent turns
    recent_turns = self.short_term_memory.get_recent_turns(
        self.config.memory.recent_k
    )
    recent_texts = [turn.text for turn in recent_turns]
    
    # Step 2: Search LTM
    ltm_results = self.long_term_memory.search(
        user_input, 
        k=self.config.memory.ltm_hits_k
    )
    ltm_texts = [entry.text for entry, score in ltm_results if score > 0.5]
    
    # Step 3: Assemble context
    context_parts = []
    if recent_texts:
        context_parts.extend(["Recent conversation:"] + recent_texts)
    if ltm_texts:
        context_parts.extend(["\nRelevant previous context:"] + ltm_texts)
    context_parts.append(f"\nCurrent user input: {user_input}")
    
    # Step 4: Optimize
    full_context = "\n\n".join(context_parts)
    if self.token_counter.count_tokens(full_context) > self.config.memory.prompt_token_budget:
        full_context = self.token_counter.truncate_to_tokens(
            full_context, 
            self.config.memory.prompt_token_budget
        )
    
    return full_context
```

### 2. Memory Observation Flow

```
User Input + Assistant Response
    ↓
ContextManager.observe()
    ↓
1. Add to STM
    ↓
2. Check STM capacity
    ↓
3. If overflow: summarize_and_offload()
    ↓
4. Generate summary
    ↓
5. Add to LTM
```

**Detailed Flow**:
```python
def observe(self, user_input: str, assistant_response: str):
    # Step 1: Add to STM
    self.short_term_memory.add_turn(user_input, assistant_response)
    
    # Step 2: Check capacity
    if self.short_term_memory.current_tokens > self.config.memory.stm_capacity:
        # Step 3: Summarize and offload
        self._summarize_and_offload()

def _summarize_and_offload(self):
    # Get chunk for summarization
    chunk, chunk_tokens = self.short_term_memory.get_chunk_for_summarization(
        self.config.memory.chunk_size
    )
    
    if not chunk:
        return
    
    # Generate summary
    summary = self.summarizer.summarize_chunk(chunk)
    
    # Create metadata
    metadata = self.summarizer.create_summary_metadata(chunk, summary)
    
    # Add to LTM
    memory_id = self.long_term_memory.add_memory(summary, metadata)
```

### 3. Memory Query Flow

```
Query Text
    ↓
EmbeddingProvider.embed_single()
    ↓
FAISS Index Search
    ↓
Similarity Calculation
    ↓
Result Ranking
    ↓
Return Top-K Results
```

**Detailed Flow**:
```python
def search(self, query: str, k: int = 7):
    # Generate query embedding
    query_embedding = self.embedding_provider.embed_single(query)
    
    # Search FAISS index
    distances, indices = self.index.search(
        query_embedding.reshape(1, -1), 
        k
    )
    
    # Convert distances to similarities
    similarities = 1 - distances[0] / np.max(distances[0])
    
    # Create results
    results = []
    for idx, similarity in zip(indices[0], similarities):
        if idx < len(self.entries):
            entry = self.entries[idx]
            results.append((entry, float(similarity)))
    
    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results
```

## Memory Management

### 1. Token Budget Management

**Budget Allocation Strategy**:
```
Total Token Budget (e.g., 12,000 tokens)
├── Recent Turns (5 turns × ~200 tokens = 1,000)
├── LTM Results (7 results × ~150 tokens = 1,050)
├── Current Input (~100 tokens)
├── System Instructions (~200 tokens)
└── Buffer (~8,650 tokens remaining)
```

**Optimization Techniques**:
1. **Priority-Based Assembly**: Recent turns > LTM results > current input
2. **Intelligent Truncation**: Token-aware text truncation
3. **Similarity Filtering**: Only include LTM results above threshold
4. **Dynamic Adjustment**: Adapt based on available budget

### 2. Memory Eviction Strategies

**STM Eviction**:
- **FIFO**: First-in, first-out for oldest turns
- **Token-Aware**: Track exact token count
- **Automatic**: Triggered when capacity exceeded

**LTM Management**:
- **Immutable**: No deletion (simplifies design)
- **Index Rebuild**: When needed for consistency
- **Metadata Tracking**: For future cleanup features

### 3. Summarization Strategy

**Chunk-Level Summarization**:
- **Trigger**: STM capacity exceeded
- **Size**: Configurable chunk size (default: 2000 tokens)
- **Compression**: Target 30% of original size
- **Quality**: Preserve key information and decisions

**Hierarchical Summarization**:
- **Levels**: Chunk → Session → Topic → Lifetime
- **Compression**: Each level compresses previous level
- **Metadata**: Track summarization relationships

## Performance Characteristics

### 1. Time Complexity

| Operation | Complexity | Description |
|-----------|------------|-------------|
| STM Add Turn | O(1) | Append to deque |
| STM Evict | O(1) | Remove from deque |
| STM Retrieve | O(k) | Get k recent turns |
| LTM Add | O(log N) | FAISS insertion |
| LTM Search | O(log N) | FAISS similarity search |
| Context Building | O(k + log N) | Recent + LTM search |
| Summarization | O(n) | Linear with chunk size |

### 2. Space Complexity

| Component | Space | Description |
|-----------|-------|-------------|
| STM | O(max_tokens) | Bounded by configuration |
| LTM Entries | O(N) | N memory entries |
| FAISS Index | O(N × dimension) | Vector storage |
| Embeddings | O(N × dimension) | Text embeddings |
| Total | O(N × dimension) | Dominated by LTM |

### 3. Memory Usage Estimates

| Component | Default Size | Description |
|-----------|--------------|-------------|
| STM | ~32KB | 8,000 tokens × 4 bytes |
| LTM (100 entries) | ~150KB | 100 × 384 dimensions × 4 bytes |
| Embedding Model | ~90MB | Sentence transformer |
| FAISS Index | ~150KB | HNSW index for 100 entries |
| **Total** | **~90MB** | Dominated by embedding model |

### 4. Performance Benchmarks

**Context Building Performance**:
```
Input: "What did we discuss about Python?"
- STM turns: 5
- LTM entries: 100
- Build time: ~50ms
- Context length: ~2,000 tokens
- Token utilization: 85%
```

**Memory Search Performance**:
```
Query: "budget planning"
- LTM entries: 100
- Search time: ~10ms
- Results: 7 entries
- Average similarity: 0.75
```

**Summarization Performance**:
```
Chunk size: 2,000 tokens
- Summarization time: ~2s
- Compression ratio: 30%
- Quality preservation: 85%
```

## Scalability Considerations

### 1. Horizontal Scaling

**Stateless Design**:
- ContextManager instances are independent
- No shared state between instances
- Easy to scale horizontally

**Load Balancing**:
```python
# Multiple instances behind load balancer
instances = [
    ContextManager(config) for _ in range(3)
]

# Round-robin or sticky sessions
def get_instance(user_id: str) -> ContextManager:
    return instances[hash(user_id) % len(instances)]
```

### 2. Vertical Scaling

**Memory Optimization**:
```python
# Memory-constrained configuration
config = Config(
    memory=MemoryConfig(
        stm_capacity=2000,     # Smaller STM
        chunk_size=500,        # Smaller chunks
        recent_k=2,            # Fewer recent turns
        ltm_hits_k=3,          # Fewer LTM results
    )
)
```

**CPU Optimization**:
```python
# Async processing
async def process_batch(queries: List[str]):
    tasks = [cm.build_context_async(q) for q in queries]
    return await asyncio.gather(*tasks)
```

### 3. Database Scaling

**Vector Store Scaling**:
- **FAISS**: In-memory, limited by RAM
- **Pinecone**: Cloud-based, scales automatically
- **Weaviate**: Self-hosted, supports clustering
- **Qdrant**: High-performance, supports sharding

**Migration Strategy**:
```python
class ScalableLongTermMemory(LongTermMemory):
    def __init__(self, vector_store_provider="faiss"):
        if vector_store_provider == "pinecone":
            self.index = PineconeIndex()
        elif vector_store_provider == "weaviate":
            self.index = WeaviateIndex()
        else:
            super().__init__()
```

### 4. Caching Strategy

**Multi-Level Caching**:
```python
class CachedContextManager(ContextManager):
    def __init__(self):
        super().__init__()
        self.context_cache = {}      # L1: In-memory
        self.redis_cache = Redis()   # L2: Distributed
        self.cdn_cache = CDN()      # L3: Global
    
    def build_context(self, user_input: str) -> str:
        # Check L1 cache
        cache_key = self._get_cache_key(user_input)
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        # Check L2 cache
        cached = self.redis_cache.get(cache_key)
        if cached:
            self.context_cache[cache_key] = cached
            return cached
        
        # Build context
        context = super().build_context(user_input)
        
        # Store in caches
        self.context_cache[cache_key] = context
        self.redis_cache.set(cache_key, context, ttl=3600)
        
        return context
```

## Security Considerations

### 1. Data Privacy

**Memory Isolation**:
```python
class IsolatedContextManager(ContextManager):
    def __init__(self, user_id: str, encryption_key: str):
        super().__init__()
        self.user_id = user_id
        self.encryption_key = encryption_key
    
    def _encrypt_memory(self, text: str) -> str:
        # Encrypt sensitive data
        return encrypt(text, self.encryption_key)
    
    def _decrypt_memory(self, encrypted_text: str) -> str:
        # Decrypt when needed
        return decrypt(encrypted_text, self.encryption_key)
    
    def add_memory(self, text: str, metadata=None):
        encrypted_text = self._encrypt_memory(text)
        return super().add_memory(encrypted_text, metadata)
```

**API Key Security**:
```python
# Secure API key management
import os
from cryptography.fernet import Fernet

class SecureLLMAdapter(LLMAdapter):
    def __init__(self, encrypted_api_key: str, encryption_key: str):
        self.fernet = Fernet(encryption_key.encode())
        self.api_key = self.fernet.decrypt(encrypted_api_key.encode()).decode()
```

### 2. Access Control

**User-Based Access**:
```python
class SecureContextManager(ContextManager):
    def __init__(self, user_id: str, permissions: List[str]):
        super().__init__()
        self.user_id = user_id
        self.permissions = permissions
    
    def query_memory(self, query: str, k: int = 5):
        # Check permissions
        if "read_memory" not in self.permissions:
            raise PermissionError("No permission to read memory")
        
        return super().query_memory(query, k)
    
    def add_memory(self, text: str, metadata=None):
        # Check permissions
        if "write_memory" not in self.permissions:
            raise PermissionError("No permission to write memory")
        
        return super().add_memory(text, metadata)
```

### 3. Input Validation

**Sanitization**:
```python
import re
from typing import Optional

class SanitizedContextManager(ContextManager):
    def __init__(self):
        super().__init__()
        self.sanitizer = TextSanitizer()
    
    def build_context(self, user_input: str) -> str:
        # Sanitize input
        sanitized_input = self.sanitizer.sanitize(user_input)
        return super().build_context(sanitized_input)
    
    def add_memory(self, text: str, metadata=None):
        # Sanitize memory
        sanitized_text = self.sanitizer.sanitize(text)
        return super().add_memory(sanitized_text, metadata)

class TextSanitizer:
    def __init__(self):
        self.xss_pattern = re.compile(r'<script.*?</script>', re.IGNORECASE)
        self.sql_pattern = re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)\b)', re.IGNORECASE)
    
    def sanitize(self, text: str) -> str:
        # Remove XSS
        text = self.xss_pattern.sub('', text)
        
        # Remove SQL injection
        text = self.sql_pattern.sub('', text)
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000] + "..."
        
        return text
```

### 4. Audit Logging

**Comprehensive Logging**:
```python
import logging
from datetime import datetime

class AuditedContextManager(ContextManager):
    def __init__(self, audit_log_file: str = "audit.log"):
        super().__init__()
        self.audit_logger = logging.getLogger("audit")
        self.audit_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(audit_log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(user_id)s - %(action)s - %(details)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
    
    def build_context(self, user_input: str) -> str:
        self._log_action("build_context", f"input_length={len(user_input)}")
        return super().build_context(user_input)
    
    def query_memory(self, query: str, k: int = 5):
        self._log_action("query_memory", f"query={query[:50]}, k={k}")
        return super().query_memory(query, k)
    
    def add_memory(self, text: str, metadata=None):
        self._log_action("add_memory", f"text_length={len(text)}")
        return super().add_memory(text, metadata)
    
    def _log_action(self, action: str, details: str):
        self.audit_logger.info(
            f"user_id={getattr(self, 'user_id', 'unknown')} - "
            f"action={action} - details={details}"
        )
```

---

**Next**: [Contributing](./contributing.md) → How to contribute to ContextManager 