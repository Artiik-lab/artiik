# Core Concepts

This section provides a deep understanding of ContextManager's core concepts, architecture, and how the different components work together to provide intelligent context management for AI agents.

## 🧠 Memory Architecture

ContextManager implements a **dual-memory system** inspired by human memory processes, with automatic management and optimization.

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    ContextManager                          │
├─────────────────────────────────────────────────────────────┤
│  Short-Term Memory (STM)    │  Long-Term Memory (LTM)    │
│  ┌─────────────────────┐    │  ┌─────────────────────┐    │
│  │ Recent Turns        │    │  │ Vector Store        │    │
│  │ Token-aware         │    │  │ Semantic Search     │    │
│  │ Fast Access         │    │  │ Hierarchical Sums   │    │
│  │ Auto-eviction       │    │  │ Persistent          │    │
│  └─────────────────────┘    │  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Short-Term Memory (STM)

**Purpose**: Store recent conversation turns for immediate context access.

**Characteristics**:
- **Token-aware**: Tracks exact token count of each turn
- **Bounded capacity**: Configurable maximum tokens (default: 8,000)
- **Fast access**: O(1) append and retrieval operations
- **Automatic eviction**: Oldest turns removed when capacity exceeded

**Data Structure**:
```python
@dataclass
class Turn:
    user_input: str
    assistant_response: str
    token_count: int
    timestamp: float
```

**Eviction Process**:
```python
def _evict_if_needed(self):
    while self.current_tokens > self.max_tokens and self.turns:
        oldest_turn = self.turns.popleft()
        self.current_tokens -= oldest_turn.token_count
```

### Long-Term Memory (LTM)

**Purpose**: Store semantic representations of conversations for future retrieval.

**Characteristics**:
- **Vector-based**: FAISS index for similarity search
- **Semantic search**: Find relevant memories by meaning
- **Hierarchical summaries**: Multi-level conversation compression
- **Persistent**: Survives across sessions
- **Scoped**: Optional `session_id` and `task_id` scoping for isolation

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

**Search Process**:
```python
def search(self, query: str, k: int = 7):
    query_embedding = self.embedding_provider.embed_single(query)
    distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
    similarities = 1 - distances[0] / np.max(distances[0])
    return [(self.entries[idx], similarity) for idx, similarity in zip(indices[0], similarities)]
```

## 📝 Hierarchical Summarization

ContextManager implements a **hierarchical summarization system** that automatically compresses conversations while preserving important information.

### Summarization Levels

```
Conversation Turns (Raw)
    ↓
Chunk Summaries (2000 tokens each)
    ↓
Session Summaries (Multiple chunks)
    ↓
Topic Summaries (Related sessions)
    ↓
Lifetime Summaries (All topics)
```

### Chunk-Level Summarization

**Process**:
1. **Chunking**: Group turns into fixed-size chunks (default: 2,000 tokens)
2. **Summarization**: Use LLM to create concise summaries
3. **Storage**: Store summaries in LTM with metadata (includes optional `session_id`/`task_id`)

**Prompt Template**:
```
Summarize the following conversation chunk in a concise way, preserving key information, decisions, and context that might be important for future reference. Focus on:

1. Main topics discussed
2. Key decisions made
3. Important facts or information shared
4. User preferences or requirements mentioned
5. Any action items or next steps

Conversation chunk:
{text}

Summary:
```

**Compression Ratio**: ~30% of original size while preserving key information.

### Hierarchical Summarization

**Process**:
1. **Collection**: Gather multiple chunk summaries
2. **Combination**: Use LLM to create higher-level summaries
3. **Recursion**: Repeat until desired hierarchy level reached

**Prompt Template**:
```
Create a higher-level summary of the following conversation summaries. This should provide an overview of the entire conversation session, highlighting:

1. Overall purpose and goals
2. Major topics covered
3. Key outcomes and decisions
4. Important patterns or themes
5. Context that would be useful for future interactions

Summaries to combine:
{summaries}

Hierarchical Summary:
```

## 💰 Token Budget Management

ContextManager implements sophisticated **token budget management** to optimize context usage within LLM limits.

### Budget Allocation

```
Total Token Budget (e.g., 12,000 tokens)
├── Recent Turns (5 turns × ~200 tokens = 1,000)
├── LTM Results (7 results × ~150 tokens = 1,050)
├── Current Input (~100 tokens)
├── System Instructions (~200 tokens)
└── Buffer (~8,650 tokens remaining)
```

### Optimization Strategies

**1. Priority-Based Assembly**:
```python
def build_context(self, user_input: str) -> str:
    # 1. Get recent turns (highest priority)
    recent_turns = self.short_term_memory.get_recent_turns(self.config.memory.recent_k)
    
    # 2. Search LTM (medium priority)
    ltm_results = self.long_term_memory.search(user_input, k=self.config.memory.ltm_hits_k)
    
    # 3. Assemble context
    context_parts = [recent_texts, ltm_texts, user_input]
    full_context = "\n\n".join(context_parts)
    
    # 4. Truncate if necessary
    return self.token_counter.truncate_to_tokens(full_context, self.config.memory.prompt_token_budget)
```

**2. Hybrid Retrieval + Intelligent Truncation**:
```python
def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
    tokens = self._encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    return self._encoder.decode(truncated_tokens)
```

**3. Similarity-Based Pruning with Keyword Prefilter**:
```python
# Filter LTM results by similarity score
ltm_texts = [entry.text for entry, score in ltm_results if score > 0.5]
```

## 🔄 Context Orchestration Engine

The **Context Orchestration Engine** is the "brainstem" that coordinates all components and manages the flow of information.

### Main Flow

```
User Input
    ↓
ContextManager.observe()
    ↓
Add to STM
    ↓
Check capacity
    ↓
If overflow: summarize_and_offload()
    ↓
ContextManager.build_context()
    ↓
Retrieve recent turns
    ↓
Search LTM
    ↓
Assemble context
    ↓
Optimize tokens
    ↓
Return context
```

### Key Methods

**1. Observe Interaction**:
```python
def observe(self, user_input: str, assistant_response: str):
    # Add to short-term memory
    self.short_term_memory.add_turn(user_input, assistant_response)
    
    # Check if we need to summarize and offload
    if self.short_term_memory.current_tokens > self.config.memory.stm_capacity:
        self._summarize_and_offload()
```

**2. Build Context**:
```python
def build_context(self, user_input: str) -> str:
    # Get recent turns from STM
    recent_turns = self.short_term_memory.get_recent_turns(self.config.memory.recent_k)
    recent_texts = [turn.text for turn in recent_turns]
    
    # Search LTM for relevant memories
    ltm_results = self.long_term_memory.search(user_input, k=self.config.memory.ltm_hits_k)
    ltm_texts = [entry.text for entry, score in ltm_results if score > 0.5]
    
    # Assemble and optimize
    context_parts = [recent_texts, ltm_texts, user_input]
    full_context = "\n\n".join(context_parts)
    return self.token_counter.truncate_to_tokens(full_context, self.config.memory.prompt_token_budget)
```

**3. Summarize and Offload**:
```python
def _summarize_and_offload(self):
    # Get chunk for summarization
    chunk, chunk_tokens = self.short_term_memory.get_chunk_for_summarization(
        self.config.memory.chunk_size
    )
    
    # Generate summary
    summary = self.summarizer.summarize_chunk(chunk)
    
    # Add to long-term memory
    memory_id = self.long_term_memory.add_memory(summary, metadata)
```

## 🔍 Vector Similarity Search

ContextManager uses **FAISS** (Facebook AI Similarity Search) for efficient vector similarity search in long-term memory.

### Embedding Process

**1. Text Embedding**:
```python
def embed_single(self, text: str) -> np.ndarray:
    return self.model.encode([text])[0]
```

**2. Similarity Calculation**:
```python
def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return float(similarity)
```

### Search Process

**1. Query Embedding**:
```python
query_embedding = self.embedding_provider.embed_single(query)
```

**2. FAISS Search**:
```python
distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
```

**3. Similarity Conversion**:
```python
similarities = 1 - distances[0] / np.max(distances[0])
```

**4. Result Filtering**:
```python
results = [(self.entries[idx], similarity) for idx, similarity in zip(indices[0], similarities)]
results.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
```

## 🎯 Use Case Patterns

### Pattern 1: Long Conversations

**Problem**: Maintaining context across 100+ conversation turns.

**Solution**:
```python
# Automatic summarization prevents context overflow
for turn in long_conversation:
    cm.observe(turn.user_input, turn.assistant_response)
    # ContextManager automatically summarizes old turns
```

### Pattern 2: Multi-Topic Discussions

**Problem**: Switching between topics without losing context.

**Solution**:
```python
# Vector similarity finds relevant past discussions
context = cm.build_context("What did we discuss about Python?")
# Automatically retrieves relevant memories about Python
```

### Pattern 3: Information Retrieval

**Problem**: Finding specific information from conversation history.

**Solution**:
```python
# Semantic search across all memories
results = cm.query_memory("budget planning", k=5)
for text, score in results:
    print(f"Score {score:.2f}: {text}")
```

### Pattern 4: Tool-Using Agents

**Problem**: Agents with tools but no built-in memory.

**Solution**:
```python
class ToolAgent:
    def __init__(self):
        self.cm = ContextManager()  # Provides memory layer
        self.tools = {...}          # Agent tools
    
    def respond(self, user_input: str):
        context = self.cm.build_context(user_input)  # Memory-enhanced context
        response = call_llm_with_tools(context, self.tools)
        self.cm.observe(user_input, response)  # Update memory
        return response
```

## ⚡ Performance Characteristics

### Time Complexity

- **STM operations**: O(1) append/pop
- **LTM search**: O(log N) with FAISS HNSW
- **Context building**: O(k) where k = recent turns + LTM hits
- **Summarization**: O(n) where n = chunk size

### Space Complexity

- **STM**: O(max_tokens) - bounded by configuration
- **LTM**: O(N) where N = number of memory entries
- **Embeddings**: O(N × dimension) for FAISS index

### Memory Usage

- **Default STM**: ~8,000 tokens ≈ 32KB
- **Default LTM**: ~100 entries × 384 dimensions ≈ 150KB
- **Embedding model**: ~90MB (sentence-transformers)

## 🔧 Configuration Trade-offs

### Speed vs. Quality

```python
# Fast configuration
config = Config(
    memory=MemoryConfig(
        recent_k=3,        # Fewer recent turns
        ltm_hits_k=5,      # Fewer LTM results
        chunk_size=1000,   # Smaller chunks
    )
)

# Quality configuration
config = Config(
    memory=MemoryConfig(
        recent_k=10,       # More recent turns
        ltm_hits_k=15,     # More LTM results
        chunk_size=3000,   # Larger chunks
    )
)
```

### Memory vs. Performance

```python
# Memory-constrained
config = Config(
    memory=MemoryConfig(
        stm_capacity=4000,     # Smaller STM
        prompt_token_budget=6000,  # Smaller context
    )
)

# Performance-optimized
config = Config(
    memory=MemoryConfig(
        stm_capacity=16000,    # Larger STM
        prompt_token_budget=24000,  # Larger context
    )
)
```

## 🚨 Failure Modes and Mitigations

### 1. Embedding Model Failures

**Failure**: Model download or loading fails.

**Mitigation**:
```python
try:
    self.model = SentenceTransformer(self.model_name)
except Exception as e:
    # Fallback to smaller model
    self.model = SentenceTransformer("all-MiniLM-L6-v2")
```

### 2. LLM API Failures

**Failure**: Summarization or generation fails.

**Mitigation**:
```python
try:
    summary = self.llm_adapter.generate_sync(prompt)
except Exception as e:
    # Fallback: simple concatenation
    return f"Conversation chunk: {combined_text[:200]}..."
```

### 3. Token Budget Exceeded

**Failure**: Context exceeds token limit.

**Mitigation**:
```python
# Intelligent truncation
if self.token_counter.count_tokens(full_context) > self.config.memory.prompt_token_budget:
    full_context = self.token_counter.truncate_to_tokens(
        full_context, 
        self.config.memory.prompt_token_budget
    )
```

### 4. Memory Overflow

**Failure**: Too many memories or embeddings.

**Mitigation**:
```python
# Configurable limits
config = Config(
    memory=MemoryConfig(
        stm_capacity=8000,  # Bounded STM
        ltm_hits_k=7,       # Limited LTM results
    )
)
```

---

**Next**: [API Reference](./api_reference.md) → Complete API documentation and reference 