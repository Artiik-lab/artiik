# Troubleshooting

This guide helps you resolve common issues when using ContextManager.

## 🚨 Common Issues

### Installation Issues

#### 1. Package Not Found

```bash
# Error: Could not find a version that satisfies the requirement context-manager
# Solution: Use the correct package name
pip install artiik
```

#### 2. Import Errors

```python
# Error: ModuleNotFoundError: No module named 'context_manager'
# Solution: Install the package correctly
pip install artiik

# Verify installation
python -c "from artiik import ContextManager; print('✅ Installed successfully')"
```

#### 3. Version Conflicts

```bash
# Error: Version conflicts with existing packages
# Solution: Use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install artiik
```

### API Key Issues

#### 1. Missing API Key

```bash
# Error: Missing API key for OpenAI/Anthropic
# Solution: Set environment variable
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Or set in Python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

#### 2. Invalid API Key

```python
# Error: Invalid API key
# Solution: Verify your API key is correct and has sufficient credits
# Check your OpenAI/Anthropic dashboard for key validity
```

### Model Download Issues

#### 1. Embedding Model Download Fails

```bash
# Error: Failed to download embedding model
# Solution: Check internet connection and disk space
# The model (~90MB) will be downloaded on first use

# Manual download (if needed)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### 2. Slow Model Loading

```python
# Issue: Slow embedding model loading
# Solution: The model is cached after first use
# Subsequent runs will be faster
```

### Memory Issues

#### 1. Out of Memory

```python
# Error: Out of memory during context building
# Solution: Reduce configuration limits
from artiik import Config, MemoryConfig

config = Config(
    memory=MemoryConfig(
        stm_capacity=4000,  # Reduce from 8000
        prompt_token_budget=6000,  # Reduce from 12000
        recent_k=3,  # Reduce from 5
        ltm_hits_k=5,  # Reduce from 7
    )
)
```

#### 2. High Memory Usage

```python
# Issue: High memory usage with large datasets
# Solution: Optimize configuration
config = Config(
    memory=MemoryConfig(
        chunk_size=1000,  # Reduce chunk size
        ingestion_chunk_size=200,  # Reduce ingestion chunks
    )
)
```

### Performance Issues

#### 1. Slow Context Building

```python
# Issue: Slow context building
# Solution: Optimize configuration
config = Config(
    memory=MemoryConfig(
        recent_k=3,  # Reduce recent turns
        ltm_hits_k=5,  # Reduce LTM results
        chunk_size=1000,  # Smaller chunks
    )
)
```

#### 2. Slow Memory Queries

```python
# Issue: Slow memory queries
# Solution: Check FAISS index size and optimize
stats = cm.get_stats()
print(f"LTM entries: {stats['long_term_memory']['num_entries']}")

# If too many entries, consider:
# 1. Reducing ingestion frequency
# 2. Using more aggressive summarization
# 3. Implementing memory cleanup
```

#### 3. Slow Summarization

```python
# Issue: Slow summarization process
# Solution: Adjust summarization settings
config = Config(
    memory=MemoryConfig(
        chunk_size=1000,  # Smaller chunks for faster processing
        summary_compression_ratio=0.5,  # Less aggressive compression
    ),
    async_summarization=True,  # Enable async processing
    background_summarization=True,  # Enable background processing
)
```

### Configuration Issues

#### 1. Invalid Configuration

```python
# Error: Validation error in configuration
# Solution: Check configuration values
from artiik import Config, MemoryConfig, LLMConfig

# Ensure all values are within valid ranges
config = Config(
    memory=MemoryConfig(
        stm_capacity=8000,  # Must be positive
        chunk_size=2000,  # Must be positive
        recent_k=5,  # Must be positive
        ltm_hits_k=7,  # Must be positive
        prompt_token_budget=12000,  # Must be positive
        summary_compression_ratio=0.3,  # Must be between 0 and 1
    ),
    llm=LLMConfig(
        provider="openai",  # Must be "openai" or "anthropic"
        model="gpt-4",  # Must be valid model name
        temperature=0.7,  # Must be between 0 and 1
    )
)
```

#### 2. Dimension Mismatch

```python
# Warning: Vector store dimension mismatch
# Solution: The system auto-aligns dimensions, but you can set explicitly
config = Config(
    vector_store=VectorStoreConfig(
        dimension=384,  # Match your embedding model dimension
    )
)
```

### Debugging Issues

#### 1. Enable Debug Mode

```python
# Enable detailed logging
config = Config(debug=True, log_level="DEBUG")
cm = ContextManager(config)

# Check memory statistics
stats = cm.get_stats()
print(f"STM turns: {stats['short_term_memory']['num_turns']}")
print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
print(f"STM utilization: {stats['short_term_memory']['utilization']:.2%}")
```

#### 2. Debug Context Building

```python
# Debug context building process
debug_info = cm.debug_context_building("What did we discuss?")
print(f"Recent turns: {debug_info['recent_turns_count']}")
print(f"LTM hits: {debug_info['ltm_results_count']}")
print(f"Final tokens: {debug_info['final_context_tokens']}")
print(f"Context budget: {debug_info['context_budget']}")
```

#### 3. Check Memory Contents

```python
# Query memory to see what's stored
results = cm.query_memory("your search term", k=10)
for text, score in results:
    print(f"Score {score:.2f}: {text[:100]}...")
```

### Platform-Specific Issues

#### macOS Issues

```bash
# FAISS/NumPy warnings on macOS
# Solution: Install OpenMP
brew install libomp

# Or use stable NumPy version
pip install numpy==1.26.4
```

#### Windows Issues

```bash
# FAISS installation issues on Windows
# Solution: Use conda or pre-built wheels
conda install -c conda-forge faiss-cpu
```

#### Linux Issues

```bash
# Missing system libraries
# Solution: Install required packages
sudo apt-get install libopenblas-dev liblapack-dev
```

## 🔧 Performance Optimization

### Memory Optimization

```python
# Optimize for memory-constrained environments
config = Config(
    memory=MemoryConfig(
        stm_capacity=4000,  # Reduce STM size
        chunk_size=1000,  # Smaller chunks
        recent_k=3,  # Fewer recent turns
        ltm_hits_k=5,  # Fewer LTM results
        prompt_token_budget=6000,  # Smaller context budget
    )
)
```

### Speed Optimization

```python
# Optimize for speed
config = Config(
    memory=MemoryConfig(
        recent_k=3,  # Fewer recent turns
        ltm_hits_k=5,  # Fewer LTM results
        chunk_size=1000,  # Smaller chunks
    ),
    llm=LLMConfig(
        model="gpt-3.5-turbo",  # Faster model
        max_tokens=500,  # Shorter responses
    ),
    async_summarization=True,  # Enable async processing
)
```

### Quality Optimization

```python
# Optimize for quality
config = Config(
    memory=MemoryConfig(
        recent_k=10,  # More recent turns
        ltm_hits_k=10,  # More LTM results
        chunk_size=2000,  # Larger chunks
        prompt_token_budget=16000,  # Larger context budget
        summary_compression_ratio=0.2,  # More aggressive compression
    ),
    llm=LLMConfig(
        model="gpt-4",  # Higher quality model
        temperature=0.3,  # More focused responses
    )
)
```

## 🆘 Getting Help

### Before Asking for Help

1. **Check the logs**: Enable debug mode and check for error messages
2. **Verify configuration**: Ensure all configuration values are valid
3. **Test with minimal setup**: Try with default configuration first
4. **Check dependencies**: Ensure all required packages are installed
5. **Reproduce the issue**: Create a minimal example that reproduces the problem

### Where to Get Help

- **Documentation**: Check the [API Reference](./api_reference.md) for detailed information
- **Examples**: Look at the [Examples](./examples.md) for usage patterns
- **GitHub Issues**: [Report bugs and request features](https://github.com/BoualamHamza/Context-Manager/issues)
- **Email**: Contact boualamhamza@outlook.fr for support

### Providing Useful Information

When reporting issues, include:

1. **Error message**: The complete error message
2. **Code example**: Minimal code that reproduces the issue
3. **Configuration**: Your ContextManager configuration
4. **Environment**: Python version, OS, package versions
5. **Steps to reproduce**: Clear steps to reproduce the issue

---

**Need more help?** → [API Reference](./api_reference.md) | [Examples](./examples.md) 