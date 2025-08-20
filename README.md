# ContextManager (by Artiik)

A professional, plug-and-play memory and context orchestration layer for AI agents. ContextManager abstracts away context-window limits by automatically managing short-term memory, long-term semantic recall, and hierarchical summarization to assemble high-signal, token-budgeted prompts for your models.

## 🚀 Quick Start

```python
from artiik import ContextManager

# Initialize with default settings
cm = ContextManager()

# Your agent workflow
user_input = "Can you help me plan a 10-day trip to Japan?"
context = cm.build_context(user_input)
response = call_llm(context)  # Your LLM call
cm.observe(user_input, response)
```

## 📦 Installation

```bash
pip install artiik
```

Or install from source:

```bash
git clone https://github.com/BoualamHamza/Context-Manager.git
cd Context-Manager
pip install -e .
```

## 🧩 Key Features

- **🔧 Drop-in Integration**: Works with existing agents without architecture changes
- **🧠 Intelligent Memory**: Automatic short-term and long-term memory management
- **📝 Hierarchical Summarization**: Multi-level conversation summarization
- **🔍 Semantic Search**: Vector-based memory retrieval with FAISS
- **📥 External Indexing**: Ingest files and directories into long-term memory
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

## 🔧 Basic Configuration

```python
from artiik import Config, ContextManager

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

## 🚀 Examples

### Basic Agent Integration

```python
from artiik import ContextManager
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

### Memory Querying

```python
from artiik import ContextManager

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

### Indexing Your Data

```python
from artiik import ContextManager

cm = ContextManager()

# Ingest a single file
chunks = cm.ingest_file("docs/README.md", importance=0.8)
print(f"Ingested chunks: {chunks}")

# Ingest a directory
total = cm.ingest_directory(
    "./my_repo",
    file_types=[".py", ".md"],
    recursive=True,
    importance=0.7,
)
print(f"Total chunks ingested: {total}")
```

## 🔍 Understanding the Components

### Memory Types

**Short-Term Memory (STM):**
- Stores recent conversation turns
- Token-aware with automatic eviction
- Fast access for immediate context

**Long-Term Memory (LTM):**
- Vector-based semantic storage using FAISS
- Hierarchical summaries
- Persistent across sessions

### Context Building Process

1. **Retrieve Recent**: Get last N turns from STM
2. **Search LTM**: Find relevant memories via vector similarity
3. **Assemble**: Combine recent + relevant + current input
4. **Optimize**: Truncate to fit token budget
5. **Return**: Optimized context for LLM

## 🛠️ Configuration Options

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

```bash
# Set environment variable
export OPENAI_API_KEY="your-key"
```

### 2. Model Download Issues

The embedding model (~90MB) will be downloaded on first use. Ensure you have internet connection and sufficient disk space.

### 3. Memory Issues

```python
# Reduce configuration limits for resource-constrained environments
config = Config(
    memory=MemoryConfig(
        stm_capacity=4000,  # Reduce from 8000
        prompt_token_budget=6000,  # Reduce from 12000
    )
)
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=context_manager
```

## 📈 Performance

### Benchmarks

- **Context Building**: ~50ms for typical queries
- **Memory Search**: ~10ms for 1000 entries
- **Summarization**: ~2s for 2000 token chunks
- **Memory Usage**: ~90MB for default configuration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/BoualamHamza/Context-Manager.git
cd Context-Manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

## 📄 License

ContextManager is licensed under the MIT License. See [LICENSE](LICENSE.md) for details.

## 🆘 Support

- **Documentation**: [docs/index.md](docs/index.md)
- **Issues**: [GitHub Issues](https://github.com/BoualamHamza/Context-Manager/issues)
- **Email**: boualamhamza@outlook.fr

## 🙏 Acknowledgments

- **FAISS**: Facebook AI Similarity Search for vector operations
- **Sentence Transformers**: Hugging Face for text embeddings
- **Pydantic**: Data validation and settings management
- **Loguru**: Structured logging
- **OpenAI & Anthropic**: LLM providers

---

**Ready to get started?** → [Full Documentation](docs/index.md) 