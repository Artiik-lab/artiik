# Troubleshooting

This section provides solutions to common issues, debugging techniques, and performance optimization tips for ContextManager.

## 📚 Table of Contents

- [Common Issues](#common-issues)
- [Performance Problems](#performance-problems)
- [Memory Issues](#memory-issues)
- [API and Network Issues](#api-and-network-issues)
- [Configuration Problems](#configuration-problems)
- [Debugging Techniques](#debugging-techniques)
- [Performance Optimization](#performance-optimization)

## Common Issues

### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'context_manager'`

**Solutions**:

```bash
# Install the package
pip install context-manager

# Or install from source
git clone https://github.com/contextmanager/context-manager.git
cd context-manager
pip install -e .

# Check installation
python -c "import context_manager; print('✅ Installed successfully')"
```

**Problem**: `ModuleNotFoundError: No module named 'faiss'`

**Solutions**:

```bash
# Install FAISS
pip install faiss-cpu

# For GPU support (if available)
pip install faiss-gpu

# Alternative: Use conda
conda install -c conda-forge faiss-cpu
```

### 2. Embedding Model Download Issues

**Problem**: `Failed to load embedding model`

**Solutions**:

```python
# Check internet connection
import requests
try:
    response = requests.get("https://huggingface.co", timeout=5)
    print("✅ Internet connection available")
except:
    print("❌ No internet connection")

# Use smaller model
from context_manager.llm.embeddings import EmbeddingProvider

# Try smaller model
provider = EmbeddingProvider("paraphrase-MiniLM-L3-v2")  # 61MB instead of 90MB

# Check available models
models = provider.list_available_models()
print("Available models:", list(models.keys()))
```

**Problem**: `Out of memory` during model download

**Solutions**:

```python
# Reduce memory usage
import gc
gc.collect()

# Use smaller model
provider = EmbeddingProvider("paraphrase-MiniLM-L3-v2")

# Check available RAM
import psutil
print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
```

### 3. API Key Issues

**Problem**: `Authentication error` or `Invalid API key`

**Solutions**:

```python
# Check environment variables
import os
print("OPENAI_API_KEY:", "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set")
print("ANTHROPIC_API_KEY:", "✅ Set" if os.getenv("ANTHROPIC_API_KEY") else "❌ Not set")

# Set API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Or set in Python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

**Problem**: `Rate limit exceeded`

**Solutions**:

```python
# Implement retry logic
import time
from context_manager.llm.adapters import create_llm_adapter

class RetryLLMAdapter:
    def __init__(self, provider, api_key, max_retries=3, delay=1):
        self.adapter = create_llm_adapter(provider, api_key=api_key)
        self.max_retries = max_retries
        self.delay = delay
    
    def generate_sync(self, prompt, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return self.adapter.generate_sync(prompt, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    time.sleep(self.delay * (2 ** attempt))  # Exponential backoff
                    continue
                raise
```

### 4. Token Budget Issues

**Problem**: `Context exceeds token budget`

**Solutions**:

```python
# Reduce configuration limits
from context_manager import Config, MemoryConfig

config = Config(
    memory=MemoryConfig(
        stm_capacity=4000,        # Reduce from 8000
        prompt_token_budget=6000,  # Reduce from 12000
        recent_k=3,               # Reduce from 5
        ltm_hits_k=5,             # Reduce from 7
    )
)

# Check token usage
from context_manager.utils.token_counter import TokenCounter

counter = TokenCounter()
text = "Your long text here..."
token_count = counter.count_tokens(text)
print(f"Token count: {token_count}")

# Truncate if necessary
truncated = counter.truncate_to_tokens(text, 1000)
print(f"Truncated token count: {counter.count_tokens(truncated)}")
```

## Performance Problems

### 1. Slow Context Building

**Problem**: Context building takes too long

**Diagnosis**:

```python
import time
from context_manager import ContextManager

cm = ContextManager()

# Profile context building
start_time = time.time()
context = cm.build_context("Test input")
build_time = time.time() - start_time

print(f"Context build time: {build_time:.3f} seconds")

# Debug context building
debug_info = cm.debug_context_building("Test input")
print(f"Recent turns: {debug_info['recent_turns_count']}")
print(f"LTM hits: {debug_info['ltm_results_count']}")
print(f"Final tokens: {debug_info['final_context_tokens']}")
```

**Solutions**:

```python
# Optimize configuration for speed
config = Config(
    memory=MemoryConfig(
        recent_k=3,        # Fewer recent turns
        ltm_hits_k=5,      # Fewer LTM results
        chunk_size=1000,   # Smaller chunks
    ),
    llm=LLMConfig(
        model="gpt-3.5-turbo",  # Faster model
        max_tokens=500,          # Shorter responses
    )
)

# Use async context building
import asyncio
from context_manager import AsyncContextManager

async def fast_context_building():
    cm = AsyncContextManager()
    context = await cm.build_context_async("Test input")
    return context

# Run async
context = asyncio.run(fast_context_building())
```

### 2. Slow Memory Search

**Problem**: LTM search is slow

**Diagnosis**:

```python
import time
from context_manager.memory.long_term import LongTermMemory

ltm = LongTermMemory()

# Profile search performance
start_time = time.time()
results = ltm.search("test query", k=10)
search_time = time.time() - start_time

print(f"Search time: {search_time:.3f} seconds")
print(f"Number of entries: {len(ltm.entries)}")

# Check FAISS index
stats = ltm.get_stats()
print(f"Index size: {stats['index_size']}")
```

**Solutions**:

```python
# Optimize FAISS index
from context_manager import VectorStoreConfig

config = Config(
    vector_store=VectorStoreConfig(
        index_type="HNSW",     # Faster than "Flat"
        metric="cosine",        # Optimized for similarity
    )
)

# Reduce search scope
results = ltm.search("test query", k=5)  # Fewer results

# Use batch search for multiple queries
queries = ["query1", "query2", "query3"]
batch_results = [ltm.search(q, k=3) for q in queries]
```

### 3. High Memory Usage

**Problem**: ContextManager uses too much RAM

**Diagnosis**:

```python
import psutil
import os

# Check memory usage
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f"Memory usage: {memory_info.rss / 1024**2:.1f} MB")

# Check ContextManager memory
stats = cm.get_stats()
print(f"STM turns: {stats['short_term_memory']['num_turns']}")
print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
```

**Solutions**:

```python
# Reduce memory footprint
config = Config(
    memory=MemoryConfig(
        stm_capacity=2000,      # Smaller STM
        chunk_size=500,         # Smaller chunks
        recent_k=2,             # Fewer recent turns
        ltm_hits_k=3,           # Fewer LTM results
    )
)

# Use smaller embedding model
from context_manager.llm.embeddings import EmbeddingProvider
provider = EmbeddingProvider("paraphrase-MiniLM-L3-v2")  # 61MB vs 90MB

# Clear memory periodically
cm.clear_memory()
```

## Memory Issues

### 1. STM Overflow

**Problem**: Short-term memory exceeds capacity

**Diagnosis**:

```python
# Check STM usage
stats = cm.get_stats()
stm_stats = stats['short_term_memory']
print(f"STM turns: {stm_stats['num_turns']}")
print(f"STM tokens: {stm_stats['current_tokens']}")
print(f"STM capacity: {stm_stats['max_tokens']}")
print(f"STM utilization: {stm_stats['utilization']:.2%}")
```

**Solutions**:

```python
# Increase STM capacity
config = Config(
    memory=MemoryConfig(
        stm_capacity=12000,  # Increase from 8000
    )
)

# Reduce chunk size for more frequent summarization
config = Config(
    memory=MemoryConfig(
        chunk_size=1000,  # Reduce from 2000
    )
)

# Manually trigger summarization
if cm.short_term_memory.current_tokens > cm.config.memory.stm_capacity * 0.8:
    cm._summarize_and_offload()
```

### 2. LTM Index Issues

**Problem**: FAISS index corruption or memory issues

**Diagnosis**:

```python
# Check LTM health
stats = cm.get_stats()
ltm_stats = stats['long_term_memory']
print(f"LTM entries: {ltm_stats['num_entries']}")
print(f"Index size: {ltm_stats['index_size']}")

# Test search functionality
try:
    results = cm.long_term_memory.search("test", k=1)
    print("✅ LTM search working")
except Exception as e:
    print(f"❌ LTM search failed: {e}")
```

**Solutions**:

```python
# Rebuild FAISS index
cm.long_term_memory._rebuild_index()

# Clear and rebuild LTM
cm.long_term_memory.clear()
# Re-add important memories
for memory in important_memories:
    cm.long_term_memory.add_memory(memory)

# Use different index type
config = Config(
    vector_store=VectorStoreConfig(
        index_type="Flat",  # More stable than HNSW
    )
)
```

### 3. Memory Persistence Issues

**Problem**: Memory not persisting across sessions

**Solutions**:

```python
# Implement custom persistence
import pickle
import os

class PersistentContextManager(ContextManager):
    def __init__(self, persistence_file="memory.pkl", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistence_file = persistence_file
        self._load_memory()
    
    def _load_memory(self):
        """Load memory from file."""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                    # Restore memory state
                    self.short_term_memory = loaded_data['stm']
                    self.long_term_memory = loaded_data['ltm']
            except Exception as e:
                print(f"Failed to load memory: {e}")
    
    def _save_memory(self):
        """Save memory to file."""
        try:
            data = {
                'stm': self.short_term_memory,
                'ltm': self.long_term_memory
            }
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to save memory: {e}")
    
    def observe(self, user_input: str, assistant_response: str):
        """Observe with persistence."""
        super().observe(user_input, assistant_response)
        self._save_memory()
```

## API and Network Issues

### 1. OpenAI API Issues

**Problem**: OpenAI API errors

**Common Errors and Solutions**:

```python
# Rate limit error
try:
    response = llm_adapter.generate_sync(prompt)
except Exception as e:
    if "rate limit" in str(e).lower():
        print("Rate limit exceeded. Waiting...")
        time.sleep(60)  # Wait 1 minute
        response = llm_adapter.generate_sync(prompt)

# Authentication error
if "authentication" in str(e).lower():
    print("Check your API key")
    print("Current key:", os.getenv("OPENAI_API_KEY")[:10] + "...")

# Model not found
if "model not found" in str(e).lower():
    print("Check model name")
    print("Available models: gpt-4, gpt-3.5-turbo, etc.")
```

### 2. Anthropic API Issues

**Problem**: Anthropic API errors

**Solutions**:

```python
# Check API key format
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key.startswith("sk-ant-"):
    print("Invalid Anthropic API key format")

# Check model name
model = "claude-3-sonnet-20240229"  # Correct format
```

### 3. Network Connectivity Issues

**Problem**: Network timeouts or connection errors

**Solutions**:

```python
# Implement retry with exponential backoff
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Network error, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            raise

# Usage
response = retry_with_backoff(lambda: llm_adapter.generate_sync(prompt))
```

## Configuration Problems

### 1. Invalid Configuration

**Problem**: Configuration validation errors

**Diagnosis**:

```python
from context_manager import Config, MemoryConfig, LLMConfig

# Test configuration
try:
    config = Config(
        memory=MemoryConfig(
            stm_capacity=-1000,  # Invalid negative value
        )
    )
except Exception as e:
    print(f"Configuration error: {e}")

# Validate configuration
def validate_config(config):
    issues = []
    
    if config.memory.stm_capacity <= 0:
        issues.append("STM capacity must be positive")
    
    if config.memory.prompt_token_budget <= 0:
        issues.append("Prompt token budget must be positive")
    
    if config.memory.recent_k <= 0:
        issues.append("Recent k must be positive")
    
    if config.memory.ltm_hits_k <= 0:
        issues.append("LTM hits k must be positive")
    
    return issues

# Check configuration
config = Config()
issues = validate_config(config)
if issues:
    print("Configuration issues:", issues)
else:
    print("✅ Configuration is valid")
```

**Solutions**:

```python
# Fix common configuration issues
config = Config(
    memory=MemoryConfig(
        stm_capacity=8000,          # Must be positive
        prompt_token_budget=12000,  # Must be positive
        recent_k=5,                 # Must be positive
        ltm_hits_k=7,              # Must be positive
        chunk_size=2000,            # Must be positive
    ),
    llm=LLMConfig(
        provider="openai",          # Valid provider
        model="gpt-4",             # Valid model
        max_tokens=1000,           # Must be positive
        temperature=0.7,           # Must be between 0 and 1
    ),
    vector_store=VectorStoreConfig(
        provider="faiss",          # Valid provider
        dimension=384,             # Must be positive
        index_type="HNSW",        # Valid index type
        metric="cosine",           # Valid metric
    )
)
```

### 2. Configuration Conflicts

**Problem**: Conflicting configuration values

**Solutions**:

```python
# Ensure configuration consistency
def validate_config_consistency(config):
    issues = []
    
    # Check token budget relationships
    if config.memory.prompt_token_budget < config.memory.stm_capacity:
        issues.append("Prompt token budget should be >= STM capacity")
    
    if config.memory.chunk_size > config.memory.stm_capacity:
        issues.append("Chunk size should be <= STM capacity")
    
    # Check embedding dimension
    if config.vector_store.dimension <= 0:
        issues.append("Embedding dimension must be positive")
    
    return issues

# Auto-fix common issues
def auto_fix_config(config):
    if config.memory.prompt_token_budget < config.memory.stm_capacity:
        config.memory.prompt_token_budget = config.memory.stm_capacity * 1.5
    
    if config.memory.chunk_size > config.memory.stm_capacity:
        config.memory.chunk_size = config.memory.stm_capacity // 2
    
    return config
```

## Debugging Techniques

### 1. Enable Debug Logging

```python
import logging
from context_manager import Config

# Enable debug logging
config = Config(
    debug=True,
    log_level="DEBUG"
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

cm = ContextManager(config)
```

### 2. Debug Context Building

```python
# Debug context building process
debug_info = cm.debug_context_building("What did we discuss about Python?")

print("🔍 Context Building Debug:")
print(f"User input: {debug_info['user_input']}")
print(f"Recent turns: {debug_info['recent_turns_count']}")
print(f"LTM hits: {debug_info['ltm_results_count']}")
print(f"Final context length: {debug_info['final_context_length']}")
print(f"Final context tokens: {debug_info['final_context_tokens']}")
print(f"Context budget: {debug_info['context_budget']}")
print(f"Token utilization: {debug_info['final_context_tokens'] / debug_info['context_budget']:.2%}")

# Show recent turns
if debug_info['recent_texts']:
    print("\nRecent turns:")
    for i, text in enumerate(debug_info['recent_texts']):
        print(f"  {i+1}. {text[:100]}...")

# Show LTM results
if debug_info['ltm_results']:
    print("\nLTM results:")
    for i, (text, score) in enumerate(debug_info['ltm_results']):
        print(f"  {i+1}. Score {score:.2f}: {text[:100]}...")
```

### 3. Memory Analysis

```python
# Analyze memory usage
def analyze_memory(cm):
    stats = cm.get_stats()
    
    print("📊 Memory Analysis:")
    print(f"STM turns: {stats['short_term_memory']['num_turns']}")
    print(f"STM tokens: {stats['short_term_memory']['current_tokens']}")
    print(f"STM utilization: {stats['short_term_memory']['utilization']:.2%}")
    print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
    print(f"LTM index size: {stats['long_term_memory']['index_size']}")
    
    # Analyze STM content
    if cm.short_term_memory.turns:
        print("\nSTM Content:")
        for i, turn in enumerate(cm.short_term_memory.turns[-3:]):  # Last 3 turns
            print(f"  Turn {i+1}: {turn.user_input[:50]}...")
    
    # Analyze LTM content
    if cm.long_term_memory.entries:
        print("\nLTM Content (sample):")
        for entry in cm.long_term_memory.entries[-3:]:  # Last 3 entries
            print(f"  Entry: {entry.text[:100]}...")

analyze_memory(cm)
```

### 4. Performance Profiling

```python
import time
import cProfile
import pstats

# Profile context building
def profile_context_building(cm, user_input):
    profiler = cProfile.Profile()
    profiler.enable()
    
    context = cm.build_context(user_input)
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
    
    return context

# Profile memory operations
def profile_memory_operations(cm):
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Add some turns
    for i in range(10):
        cm.observe(f"User input {i}", f"Assistant response {i}")
    
    # Query memory
    results = cm.query_memory("test query")
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# Run profiling
profile_context_building(cm, "What did we discuss?")
profile_memory_operations(cm)
```

## Performance Optimization

### 1. Optimize for Speed

```python
# Fast configuration
fast_config = Config(
    memory=MemoryConfig(
        stm_capacity=4000,     # Smaller STM
        chunk_size=1000,       # Smaller chunks
        recent_k=3,            # Fewer recent turns
        ltm_hits_k=5,          # Fewer LTM results
        prompt_token_budget=8000,  # Smaller context
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",  # Faster model
        max_tokens=500,          # Shorter responses
        temperature=0.3,         # More focused
    ),
    vector_store=VectorStoreConfig(
        index_type="HNSW",      # Faster than "Flat"
        metric="cosine",         # Optimized metric
    )
)
```

### 2. Optimize for Quality

```python
# Quality configuration
quality_config = Config(
    memory=MemoryConfig(
        stm_capacity=16000,    # Larger STM
        chunk_size=4000,       # Larger chunks
        recent_k=10,           # More recent turns
        ltm_hits_k=15,         # More LTM results
        prompt_token_budget=24000,  # Larger context
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",         # Best model
        max_tokens=2000,       # Longer responses
        temperature=0.7,       # More creative
    ),
    vector_store=VectorStoreConfig(
        index_type="Flat",     # More accurate than HNSW
        metric="cosine",
    )
)
```

### 3. Optimize for Memory Usage

```python
# Memory-constrained configuration
memory_config = Config(
    memory=MemoryConfig(
        stm_capacity=2000,     # Very small STM
        chunk_size=500,        # Small chunks
        recent_k=2,            # Few recent turns
        ltm_hits_k=3,          # Few LTM results
        prompt_token_budget=4000,  # Small context
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        max_tokens=300,        # Short responses
        temperature=0.5,
    ),
    vector_store=VectorStoreConfig(
        dimension=384,         # Smaller embeddings
        index_type="Flat",     # Less memory than HNSW
    )
)
```

### 4. Async Optimization

```python
import asyncio
from context_manager import AsyncContextManager

async def optimized_usage():
    cm = AsyncContextManager()
    
    # Parallel context building
    tasks = [
        cm.build_context_async(f"Query {i}")
        for i in range(5)
    ]
    
    contexts = await asyncio.gather(*tasks)
    
    # Parallel memory queries
    query_tasks = [
        cm.query_memory(f"query {i}")
        for i in range(3)
    ]
    
    results = await asyncio.gather(*query_tasks)
    
    return contexts, results

# Run optimized usage
contexts, results = asyncio.run(optimized_usage())
```

### 5. Caching Optimization

```python
from functools import lru_cache
import hashlib

class CachedContextManager(ContextManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_cache = {}
    
    def _get_cache_key(self, user_input: str) -> str:
        """Generate cache key for user input."""
        # Include recent context in cache key
        recent_turns = self.short_term_memory.get_recent_turns(3)
        context_str = user_input + "".join([turn.text for turn in recent_turns])
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def build_context(self, user_input: str) -> str:
        """Build context with caching."""
        cache_key = self._get_cache_key(user_input)
        
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        context = super().build_context(user_input)
        self.context_cache[cache_key] = context
        
        # Limit cache size
        if len(self.context_cache) > 100:
            # Remove oldest entries
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]
        
        return context

# Usage
cached_cm = CachedContextManager()
context1 = cached_cm.build_context("What is Python?")
context2 = cached_cm.build_context("What is Python?")  # Cached
print(f"Contexts equal: {context1 == context2}")
```

---

**Next**: [Architecture](./architecture.md) → System design and component interactions 