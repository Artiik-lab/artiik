# Core Concepts

This guide explains the fundamental concepts behind ContextManager and how it manages memory and context for AI agents.

## 🧠 Memory Architecture

ContextManager uses a dual-memory architecture inspired by human memory systems:

### Short-Term Memory (STM)

**Purpose**: Stores recent conversation turns for immediate access.

**Characteristics**:
- **Fast Access**: Recent turns are immediately available
- **Token-Aware**: Automatically manages memory within token limits
- **Automatic Eviction**: Oldest turns are summarized and moved to LTM when capacity is reached
- **Configurable**: Size and behavior can be tuned via `MemoryConfig.stm_capacity`

**Usage**:
```python
# Recent turns are automatically included in context
context = cm.build_context("What did we just discuss?")
# STM provides immediate access to recent conversation
```

### Long-Term Memory (LTM)

**Purpose**: Stores semantic memories and summaries for long-term recall.

**Characteristics**:
- **Vector-Based**: Uses FAISS for efficient similarity search
- **Semantic Search**: Finds relevant memories based on meaning, not just keywords
- **Persistent**: Survives across sessions and can be saved to disk
- **Hierarchical**: Stores both original memories and summaries

**Usage**:
```python
# Query for relevant memories
results = cm.query_memory("Python programming", k=5)
# LTM finds semantically similar memories
```

## 🔄 Context Building Process

ContextManager follows a systematic process to build optimized context:

### 1. Retrieve Recent Context

```python
# Get last N turns from STM
recent_turns = stm.get_recent_turns(config.memory.recent_k)
```

**Purpose**: Ensure immediate context is always available.

### 2. Search Long-Term Memory

```python
# Find relevant memories via vector similarity
ltm_results = ltm.search(user_input, k=config.memory.ltm_hits_k)
```

**Purpose**: Retrieve semantically relevant historical context.

### 3. Assemble Context

```python
# Combine recent + relevant + current input
context = assemble_context(recent_turns, ltm_results, user_input)
```

**Purpose**: Create comprehensive context from multiple sources.

### 4. Optimize for Token Budget

```python
# Truncate to fit within budget
final_context = truncate_to_budget(context, config.memory.prompt_token_budget)
```

**Purpose**: Ensure context fits within LLM token limits.

## 📝 Hierarchical Summarization

When STM capacity is exceeded, ContextManager automatically summarizes and offloads memories:

### Summarization Process

1. **Chunk Selection**: Oldest turns are grouped into chunks based on `chunk_size`
2. **LLM Summarization**: Each chunk is summarized using the configured LLM
3. **Metadata Creation**: Summary includes metadata about original turns
4. **LTM Storage**: Summaries are stored in LTM with vector embeddings

### Example

```python
# Original conversation (8 turns, 6000 tokens)
conversation = [
    "User: Hello!", "Assistant: Hi there!",
    "User: How are you?", "Assistant: I'm doing well!",
    "User: What's the weather?", "Assistant: I can't check the weather.",
    "User: Tell me about Python", "Assistant: Python is a programming language..."
]

# After summarization (1 summary, 200 tokens)
summary = "User greeted assistant, asked about weather and Python programming. Assistant responded positively and explained Python basics."
```

## 🔍 Vector Similarity Search

LTM uses FAISS for efficient vector similarity search:

### Embedding Generation

```python
# Text is converted to vector embeddings
embedding = embedding_provider.get_embedding("Python programming")
# Default: 384-dimensional vectors using sentence-transformers
```

### Similarity Search

```python
# Query is embedded and compared to stored embeddings
query_embedding = embedding_provider.get_embedding("programming languages")
results = faiss_index.search(query_embedding, k=5)
# Returns most similar memories with similarity scores
```

### Ranking Factors

Results are ranked using configurable weights:

```python
# Final score = similarity_weight * similarity + 
#               recency_weight * recency_decay + 
#               importance_weight * importance
```

## 💰 Token Budget Management

ContextManager implements intelligent token budget management:

### Budget Allocation

```python
# Example budget breakdown
config = MemoryConfig(
    stm_capacity=8000,          # STM limit
    prompt_token_budget=12000,  # Final context limit
    recent_k=5,                 # Recent turns
    ltm_hits_k=7,              # LTM results
)
```

### Optimization Strategy

1. **LTM Results**: Drop least relevant LTM hits first
2. **Recent Turns**: Drop oldest recent turns if needed
3. **Hard Truncation**: Token-level truncation as last resort

### Example Budget Usage

```python
# Typical budget breakdown
recent_context = 2000 tokens    # Last 5 turns
ltm_context = 5000 tokens       # 7 most relevant memories
user_input = 500 tokens         # Current input
total = 7500 tokens            # Within 12000 budget
```

## 🎯 Memory Scoping

ContextManager supports memory isolation through session and task scoping:

### Session Scoping

```python
# Each user gets isolated memory
user1_cm = ContextManager(session_id="user1")
user2_cm = ContextManager(session_id="user2")

# Memories are isolated by default
user1_cm.observe("I like Python", "Python is great!")
user2_cm.observe("I like JavaScript", "JavaScript is awesome!")

# Queries only return memories from the same session
```

### Task Scoping

```python
# Different tasks within the same session
planning_cm = ContextManager(session_id="user1", task_id="vacation_planning")
coding_cm = ContextManager(session_id="user1", task_id="python_coding")

# Memories are isolated by task
planning_cm.observe("I want to visit Japan", "Japan is beautiful!")
coding_cm.observe("I need to fix this bug", "Let's debug this together!")
```

### Cross-Scope Access

```python
# Enable cross-scope access when needed
shared_cm = ContextManager(
    session_id="shared",
    allow_cross_session=True,
    allow_cross_task=True
)
```

## 📊 Memory Persistence

LTM can be persisted to disk for long-term storage:

### Save Memory

```python
# Save to directory
cm.save_memory("./memory_backup")
# Creates: memory_backup/index.faiss, memory_backup/entries.json
```

### Load Memory

```python
# Load from directory
new_cm = ContextManager()
new_cm.load_memory("./memory_backup")
# Memory is restored with all embeddings and metadata
```

## 🔧 Configuration Philosophy

ContextManager follows a "configure for your use case" philosophy:

### Performance vs Quality Trade-offs

```python
# Fast configuration
fast_config = MemoryConfig(
    recent_k=3,        # Fewer recent turns
    ltm_hits_k=5,      # Fewer LTM results
    chunk_size=1000,   # Smaller chunks
)

# Quality configuration
quality_config = MemoryConfig(
    recent_k=10,       # More recent turns
    ltm_hits_k=10,     # More LTM results
    chunk_size=2000,   # Larger chunks
    prompt_token_budget=16000,  # Larger context
)
```

### Memory vs Speed Trade-offs

```python
# Memory-constrained
constrained_config = MemoryConfig(
    stm_capacity=4000,  # Smaller STM
    prompt_token_budget=6000,  # Smaller context
)

# Speed-optimized
speed_config = MemoryConfig(
    recent_k=3,        # Fewer recent turns
    ltm_hits_k=5,      # Fewer LTM results
)
```

## 🎯 Use Case Patterns

### Long Conversations

```python
# ContextManager automatically handles long conversations
for i in range(100):
    cm.observe(f"Turn {i}", f"Response {i}")
    
# Context remains relevant through summarization
context = cm.build_context("What did we discuss earlier?")
```

### Multi-Topic Discussions

```python
# Seamless topic switching
cm.observe("Let's talk about Python", "Python is great!")
cm.observe("What about cooking?", "Cooking is fun!")
cm.observe("Back to Python - what are decorators?", "Decorators are...")

# Context includes relevant information from both topics
```

### Information Retrieval

```python
# Query specific information
results = cm.query_memory("budget planning", k=3)
# Returns relevant memories even from long conversations
```

## 🔍 Debugging and Monitoring

### Memory Statistics

```python
stats = cm.get_stats()
print(f"STM turns: {stats['short_term_memory']['num_turns']}")
print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
print(f"STM utilization: {stats['short_term_memory']['utilization']:.2%}")
```

### Context Building Debug

```python
debug_info = cm.debug_context_building("What did we discuss?")
print(f"Recent turns: {debug_info['recent_turns_count']}")
print(f"LTM hits: {debug_info['ltm_results_count']}")
print(f"Final tokens: {debug_info['final_context_tokens']}")
```

---

**Ready to dive deeper?** → [API Reference](./api_reference.md) | **See examples?** → [Examples](./examples.md) 