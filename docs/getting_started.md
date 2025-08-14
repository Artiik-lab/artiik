# Getting Started

This guide will help you get up and running with ContextManager quickly. You'll learn how to install the library, configure it for your needs, and integrate it with your AI agents.

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for first-time model downloads)

### Install ContextManager

```bash
pip install context-manager
```

### Install from Source

```bash
git clone https://github.com/contextmanager/context-manager.git
cd context-manager
pip install -e .
```

### Verify Installation

```python
from context_manager import ContextManager
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
from context_manager import ContextManager

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
from context_manager import Config, ContextManager

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

## 🚀 Quick Examples

### Example 1: Simple Agent Integration

```python
from context_manager import ContextManager
import openai

# Initialize
cm = ContextManager()
openai.api_key = "your-api-key"

def simple_agent(user_input: str) -> str:
    # Build context
    context = cm.build_context(user_input)
    
    # Call LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": context}],
        max_tokens=500
    )
    
    # Get response
    assistant_response = response.choices[0].message.content
    
    # Observe interaction
    cm.observe(user_input, assistant_response)
    
    return assistant_response

# Usage
response = simple_agent("Tell me about Python programming")
print(response)
```

### Example 2: Tool-Using Agent

```python
from context_manager import ContextManager
from context_manager.llm.adapters import create_llm_adapter

class ToolAgent:
    def __init__(self):
        self.cm = ContextManager()
        self.llm = create_llm_adapter("openai", api_key="your-key")
        self.tools = {
            "search": self._search_web,
            "calculate": self._calculate,
            "get_weather": self._get_weather
        }
    
    def _search_web(self, query: str) -> str:
        return f"Search results for: {query}"
    
    def _calculate(self, expression: str) -> str:
        try:
            return f"Result: {eval(expression)}"
        except:
            return "Error: Invalid expression"
    
    def _get_weather(self, location: str) -> str:
        return f"Weather in {location}: 72°F, sunny"
    
    def respond(self, user_input: str) -> str:
        # Build context with tool information
        context = self.cm.build_context(user_input)
        tool_prompt = f"""
You have access to these tools:
- search: Search the web
- calculate: Perform calculations
- get_weather: Get weather information

User: {context}
"""
        
        # Generate response
        response = self.llm.generate_sync(tool_prompt)
        
        # Observe interaction
        self.cm.observe(user_input, response)
        
        return response

# Usage
agent = ToolAgent()
response = agent.respond("What's 15 * 23?")
print(response)
```

### Example 3: Memory Querying

```python
from context_manager import ContextManager

cm = ContextManager()

# Add some conversation history
conversation = [
    ("I'm planning a trip to Japan", "That sounds exciting!"),
    ("I want to visit Tokyo and Kyoto", "Great choices! Tokyo is modern, Kyoto is traditional."),
    ("What's the best time to visit?", "Spring for cherry blossoms or fall for autumn colors."),
    ("How much should I budget?", "Around $200-300 per day for a comfortable trip.")
]

for user_input, response in conversation:
    cm.observe(user_input, response)

# Query memory
results = cm.query_memory("Japan budget", k=3)
for text, score in results:
    print(f"Score {score:.2f}: {text}")
```

## 🔍 Understanding the Components

### Memory Types

**Short-Term Memory (STM):**
- Stores recent conversation turns
- Token-aware with automatic eviction
- Fast access for immediate context

**Long-Term Memory (LTM):**
- Vector-based semantic storage
- Hierarchical summaries
- Persistent across sessions

### Context Building Process

1. **Retrieve Recent**: Get last N turns from STM
2. **Search LTM**: Find relevant memories via vector similarity
3. **Assemble**: Combine recent + relevant + current input
4. **Optimize**: Truncate to fit token budget
5. **Return**: Optimized context for LLM

### Token Budget Management

```python
# Example token budget breakdown
config = Config(
    memory=MemoryConfig(
        stm_capacity=8000,          # STM limit
        prompt_token_budget=12000,  # Final context limit
        recent_k=5,                 # Recent turns
        ltm_hits_k=7,              # LTM results
    )
)
```

## 🛠️ Configuration Options

### Memory Configuration

```python
from context_manager import MemoryConfig

memory_config = MemoryConfig(
    stm_capacity=8000,              # Max tokens in short-term memory
    chunk_size=2000,                # Tokens per summarization chunk
    recent_k=5,                     # Recent turns always in context
    ltm_hits_k=7,                   # Number of LTM results to retrieve
    prompt_token_budget=12000,      # Max tokens for final context
    summary_compression_ratio=0.3,  # Summary compression target
)
```

### LLM Configuration

```python
from context_manager import LLMConfig

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
from context_manager import VectorStoreConfig

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

## 🚨 Common Issues

### 1. API Key Issues

```python
# Error: Missing API key
# Solution: Set environment variable
export OPENAI_API_KEY="your-key"
```

### 2. Model Download Issues

```python
# Error: Failed to load embedding model
# Solution: Check internet connection and disk space
# The model (~90MB) will be downloaded on first use
```

### 3. Memory Issues

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

### 4. Performance Issues

```python
# Slow context building
# Solution: Adjust configuration
config = Config(
    memory=MemoryConfig(
        recent_k=3,        # Reduce from 5
        ltm_hits_k=5,      # Reduce from 7
    )
)
```

## 📈 Next Steps

Now that you have ContextManager set up, explore:

- **[Core Concepts](./core_concepts.md)**: Deep dive into memory architecture
- **[API Reference](./api_reference.md)**: Complete API documentation
- **[Examples](./examples.md)**: Advanced usage patterns
- **[Advanced Usage](./advanced_usage.md)**: Custom implementations

---

**Next**: [Core Concepts](./core_concepts.md) → Understand the memory architecture and concepts 