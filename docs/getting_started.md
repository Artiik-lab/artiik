# Getting Started

This guide will help you get up and running with ContextManager quickly. You'll learn how to install the library, configure it for your needs, and integrate it with your AI agents.

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for first-time model downloads)

### Install ContextManager

```bash
pip install artiik
```

### Install from Source

```bash
git clone https://github.com/BoualamHamza/Context-Manager.git
cd Context-Manager
pip install -e .
```

### Verify Installation

```python
from artiik import ContextManager
print("✅ ContextManager installed successfully!")
```

## 🔧 Basic Setup

### 1. Environment Setup

Set your API keys as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. Basic Usage

Here's the simplest way to use ContextManager:

```python
from artiik import ContextManager

# Initialize with default settings
cm = ContextManager()

# Your agent workflow
user_input = "Hello! I'm planning a trip to Japan."
context = cm.build_context(user_input)
response = call_llm(context)  # Your LLM call
cm.observe(user_input, response)
```

### 3. Configuration

Customize ContextManager for your needs:

```python
from artiik import Config, ContextManager

# Create custom configuration
config = Config(
    memory=MemoryConfig(
        stm_capacity=8000,          # Short-term memory limit
        chunk_size=2000,            # Summarization chunk size
        recent_k=5,                 # Recent turns to keep
        ltm_hits_k=7,               # Long-term memory results
        prompt_token_budget=12000,  # Final context limit
    ),
    llm=LLMConfig(
        provider="openai",          # or "anthropic"
        model="gpt-4",             # Model name
        api_key="your-api-key",    # API key
        max_tokens=1000,           # Response limit
        temperature=0.7,           # Creativity level
    ),
    vector_store=VectorStoreConfig(
        provider="faiss",          # Vector database
        dimension=384,             # Embedding dimension
        index_type="HNSW",        # Index type
        metric="cosine",           # Similarity metric
    ),
    debug=True,                    # Enable debug logging
    log_level="INFO"              # Logging level
)

# Initialize with custom config
cm = ContextManager(config)
```

## 🚀 Your First Integration

### Simple Agent with ContextManager

```python
from artiik import ContextManager
import openai

# Initialize
cm = ContextManager()
openai.api_key = "your-api-key"

def simple_agent(user_input: str) -> str:
    # Build context from memory
    context = cm.build_context(user_input)
    
    # Call your LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": context}],
        max_tokens=500
    )
    
    # Get response
    assistant_response = response.choices[0].message.content
    
    # Store the interaction in memory
    cm.observe(user_input, assistant_response)
    
    return assistant_response

# Use the agent
response = simple_agent("Hello! I'm planning a trip to Japan.")
print(response)
```

### Memory Querying

```python
from artiik import ContextManager

cm = ContextManager()

# Add some conversation history
conversation = [
    ("I'm planning a trip to Japan", "That sounds exciting!"),
    ("I want to visit Tokyo and Kyoto", "Great choices!"),
    ("What's the best time to visit?", "Spring for cherry blossoms!"),
    ("How much should I budget?", "Around $200-300 per day.")
]

for user_input, response in conversation:
    cm.observe(user_input, response)

# Query memory for specific information
results = cm.query_memory("Japan budget", k=3)
for text, score in results:
    print(f"Score {score:.2f}: {text}")
```

### Indexing External Data

```python
from artiik import ContextManager

cm = ContextManager()

# Ingest a single file
chunks = cm.ingest_file("docs/README.md", importance=0.8)
print(f"Ingested {chunks} chunks from README.md")

# Ingest a directory
total = cm.ingest_directory(
    "./my_repo",
    file_types=[".py", ".md"],
    recursive=True,
    importance=0.7,
)
print(f"Total chunks ingested: {total}")

# Now you can ask questions about your indexed data
context = cm.build_context("Where is authentication handled?")
```

## 🔍 Understanding the Workflow

### The ContextManager Workflow

1. **Initialize**: Create a ContextManager instance with your configuration
2. **Build Context**: Call `build_context(user_input)` to get optimized context
3. **Call LLM**: Use the context with your language model
4. **Observe**: Call `observe(user_input, response)` to store the interaction
5. **Repeat**: The cycle continues, building richer context over time

### Memory Types

**Short-Term Memory (STM):**
- Stores recent conversation turns
- Automatically managed within token limits
- Fast access for immediate context

**Long-Term Memory (LTM):**
- Vector-based semantic storage using FAISS
- Stores summaries and external data
- Persistent across sessions

### Context Building Process

1. **Retrieve Recent**: Get last N turns from STM
2. **Search LTM**: Find relevant memories via vector similarity
3. **Assemble**: Combine recent + relevant + current input
4. **Optimize**: Truncate to fit token budget
5. **Return**: Optimized context for LLM

## 🛠️ Configuration Deep Dive

### Memory Configuration

```python
from artiik import MemoryConfig

memory_config = MemoryConfig(
    stm_capacity=8000,              # Max tokens in short-term memory
    chunk_size=2000,                # Tokens per summarization chunk
    recent_k=5,                     # Recent turns always in context
    ltm_hits_k=7,                   # Number of LTM results to retrieve
    prompt_token_budget=12000,      # Max tokens for final context
    summary_compression_ratio=0.3,  # Summary compression target
    # Ingestion settings
    ingestion_chunk_size=400,       # Tokens per ingestion chunk
    ingestion_chunk_overlap=50,     # Token overlap between chunks
    # Ranking weights
    similarity_weight=1.0,          # Weight for vector similarity
    recency_weight=0.0,             # Weight for recency
    importance_weight=0.0,          # Weight for importance
    recency_half_life_seconds=604800.0,  # Half-life for recency decay (7 days)
)
```

### LLM Configuration

```python
from artiik import LLMConfig

llm_config = LLMConfig(
    provider="openai",              # "openai" or "anthropic"
    model="gpt-4",                 # Model name
    api_key="your-api-key",        # API key
    max_tokens=1000,               # Response token limit
    temperature=0.7,               # Creativity (0.0-1.0)
)
```

### Vector Store Configuration

```python
from artiik import VectorStoreConfig

vector_config = VectorStoreConfig(
    provider="faiss",              # Vector database provider
    dimension=384,                 # Embedding dimension
    index_type="HNSW",            # "HNSW" or "Flat"
    metric="cosine",               # Similarity metric
)
```

## 📊 Monitoring and Debugging

### Enable Debug Mode

```python
config = Config(debug=True, log_level="DEBUG")
cm = ContextManager(config)
```

### Get Memory Statistics

```python
stats = cm.get_stats()
print(f"STM turns: {stats['short_term_memory']['num_turns']}")
print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
print(f"STM utilization: {stats['short_term_memory']['utilization']:.2%}")
```

### Debug Context Building

```python
debug_info = cm.debug_context_building("What did we discuss?")
print(f"Recent turns: {debug_info['recent_turns_count']}")
print(f"LTM hits: {debug_info['ltm_results_count']}")
print(f"Final tokens: {debug_info['final_context_tokens']}")
```

## 🚨 Common Setup Issues

### 1. Missing API Key

```bash
# Error: Missing API key
# Solution: Set environment variable
export OPENAI_API_KEY="your-key"
```

### 2. Model Download Issues

```bash
# Error: Failed to load embedding model
# Solution: Check internet connection and disk space
# The model (~90MB) will be downloaded on first use
```

### 3. Import Errors

```python
# Error: ModuleNotFoundError: No module named 'context_manager'
# Solution: Install the package correctly
pip install artiik
```

### 4. Memory Issues

```python
# Error: Out of memory
# Solution: Reduce configuration limits
config = Config(
    memory=MemoryConfig(
        stm_capacity=4000,  # Reduce from 8000
        prompt_token_budget=6000,  # Reduce from 12000
    )
)
```

## 📈 Next Steps

Now that you have ContextManager set up, explore:

- **[Core Concepts](./core_concepts.md)**: Deep dive into memory architecture
- **[API Reference](./api_reference.md)**: Complete API documentation
- **[Examples](./examples.md)**: Advanced usage patterns
- **[Advanced Usage](./advanced_usage.md)**: Custom implementations
- **[Troubleshooting](./troubleshooting.md)**: Common issues and solutions

## 🧪 Testing Your Setup

Run the demo to test your installation:

```bash
# From the project root
python demo.py
```

Or run individual demos:

```bash
python demos/demo_basic_chat.py
python demos/demo_openai_chat.py
```

---

**Ready to dive deeper?** → [Core Concepts](./core_concepts.md) 