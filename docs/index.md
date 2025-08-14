# ContextManager Documentation

Welcome to the comprehensive documentation for ContextManager - a modular, plug-and-play memory and context management layer for AI agents.

## 🚀 Quick Start

Get up and running in minutes:

```python
from context_manager import ContextManager

# Initialize with default settings
cm = ContextManager()

# Your agent workflow
user_input = "Can you help me plan a 10-day trip to Japan?"
context = cm.build_context(user_input)
response = call_llm(context)  # Your LLM call
cm.observe(user_input, response)
```

## 📚 Documentation Sections

### [Getting Started](./getting_started.md)
**Perfect for newcomers** - Installation, basic setup, and your first ContextManager integration.

- Installation and environment setup
- Basic usage patterns
- Configuration options
- First steps with ContextManager
- Common setup issues and solutions

### [Core Concepts](./core_concepts.md)
**Essential reading** - Deep understanding of how ContextManager works.

- Memory architecture overview
- Short-term vs Long-term memory
- Hierarchical summarization
- Token budget management
- Context orchestration
- Vector similarity search
- Performance characteristics

### [API Reference](./api_reference.md)
**Complete reference** - Every class, method, and configuration option.

- Complete API documentation
- Class and method references
- Configuration options
- Error handling
- Code examples for every method

### [Examples](./examples.md)
**Practical patterns** - Real-world usage examples and integration patterns.

- Basic agent integration
- Tool-using agents
- Multi-session persistence
- Custom summarization
- Memory querying patterns
- Performance optimization examples

### [Advanced Usage](./advanced_usage.md)
**For power users** - Custom implementations and advanced patterns.

- Custom memory implementations
- Multi-agent scenarios
- Performance optimization
- Debugging and monitoring
- Integration patterns
- Production deployment

### [Troubleshooting](./troubleshooting.md)
**Problem solving** - Common issues, solutions, and debugging techniques.

- Common issues and solutions
- Performance optimization
- Debugging techniques
- Error handling patterns
- Memory management issues

### [Architecture](./architecture.md)
**System design** - Deep dive into the internal architecture and design decisions.

- System design overview
- Component interactions
- Data flow diagrams
- Performance characteristics
- Scalability considerations
- Security considerations

### [Contributing](./contributing.md)
**For contributors** - How to contribute to ContextManager development.

- Development setup
- Coding standards
- Testing guidelines
- Pull request process
- Release process

## 🧩 Key Features

- **🔧 Drop-in Integration**: Works with existing agents without architecture changes
- **🧠 Intelligent Memory**: Automatic short-term and long-term memory management
- **📝 Hierarchical Summarization**: Multi-level conversation summarization
- **🔍 Semantic Search**: Vector-based memory retrieval
- **💰 Token Optimization**: Smart context assembly within budget constraints
- **🔄 Multi-LLM Support**: OpenAI, Anthropic, and extensible adapters
- **📊 Debug Tools**: Context building visualization and monitoring
- **⚡ Performance**: Optimized for production use with configurable trade-offs

## 🎯 Use Cases

- **Long Conversations**: Maintain context across 100+ turns
- **Multi-Topic Discussions**: Seamless context switching
- **Information Retrieval**: "What did we discuss about X?" queries
- **Tool-Using Agents**: Add memory to agents with external tools
- **Multi-Session Persistence**: Context continuity across sessions
- **Resource-Constrained Environments**: Configurable memory and processing limits

## 📦 Installation

```bash
pip install context-manager
```

Or install from source:

```bash
git clone https://github.com/contextmanager/context-manager.git
cd context-manager
pip install -e .
```

## 🔧 Quick Configuration

```python
from context_manager import Config, ContextManager

# Custom configuration
config = Config(
    memory=MemoryConfig(
        stm_capacity=8000,          # Short-term memory tokens
        chunk_size=2000,            # Summarization chunk size
        recent_k=5,                 # Recent turns in context
        ltm_hits_k=7,               # Long-term memory results
        prompt_token_budget=12000,  # Final context limit
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key"
    )
)

cm = ContextManager(config)
```

## 🚀 Quick Examples

### Basic Agent Integration

```python
from context_manager import ContextManager
import openai

cm = ContextManager()
openai.api_key = "your-api-key"

def simple_agent(user_input: str) -> str:
    context = cm.build_context(user_input)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": context}],
        max_tokens=500
    )
    assistant_response = response.choices[0].message.content
    cm.observe(user_input, assistant_response)
    return assistant_response
```

### Tool-Using Agent

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
    
    def respond(self, user_input: str) -> str:
        context = self.cm.build_context(user_input)
        tool_prompt = f"""
You have access to these tools:
- search: Search the web
- calculate: Perform calculations
- get_weather: Get weather information

User: {context}
"""
        response = self.llm.generate_sync(tool_prompt)
        self.cm.observe(user_input, response)
        return response
```

### Memory Querying

```python
from context_manager import ContextManager

cm = ContextManager()

# Add conversation history
conversation = [
    ("I'm planning a trip to Japan", "That sounds exciting!"),
    ("I want to visit Tokyo and Kyoto", "Great choices!"),
    ("What's the best time to visit?", "Spring for cherry blossoms!"),
    ("How much should I budget?", "Around $200-300 per day.")
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

- **[Getting Started](./getting_started.md)**: Complete setup and first steps
- **[Core Concepts](./core_concepts.md)**: Deep dive into memory architecture
- **[API Reference](./api_reference.md)**: Complete API documentation
- **[Examples](./examples.md)**: Advanced usage patterns
- **[Advanced Usage](./advanced_usage.md)**: Custom implementations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./contributing.md) for details.

## 📄 License

ContextManager is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## 🆘 Support

- **Documentation**: This site
- **Issues**: [GitHub Issues](https://github.com/contextmanager/context-manager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/contextmanager/context-manager/discussions)
- **Email**: support@contextmanager.ai

---

**Ready to get started?** → [Getting Started](./getting_started.md) 