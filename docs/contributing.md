# Contributing to ContextManager

Thank you for your interest in contributing to ContextManager! This guide will help you get started with development, understand our coding standards, and contribute effectively.

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Code Review Guidelines](#code-review-guidelines)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda
- Basic understanding of Python, AI/ML concepts

### Fork and Clone

1. **Fork the repository**:
   - Go to [ContextManager GitHub repository](https://github.com/contextmanager/context-manager)
   - Click "Fork" in the top right

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/context-manager.git
   cd context-manager
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/contextmanager/context-manager.git
   ```

### Development Environment

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install in development mode**:
   ```bash
   pip install -e .
   ```

4. **Run tests**:
   ```bash
   pytest tests/
   ```

## Development Setup

### Project Structure

```
context-manager/
├── context_manager/          # Main package
│   ├── core/                # Core components
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration classes
│   │   └── orchestrator.py  # Main ContextManager class
│   ├── memory/              # Memory components
│   │   ├── __init__.py
│   │   ├── short_term.py    # Short-term memory
│   │   ├── long_term.py     # Long-term memory
│   │   └── summarizer.py    # Summarization
│   ├── llm/                 # LLM components
│   │   ├── __init__.py
│   │   ├── adapters.py      # LLM adapters
│   │   └── embeddings.py    # Embedding provider
│   ├── utils/               # Utilities
│   │   ├── __init__.py
│   │   └── token_counter.py # Token counting
│   └── examples/            # Example usage
│       ├── __init__.py
│       └── agent_example.py # Example agents
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_context_manager.py
│   ├── test_memory.py
│   └── test_llm.py
├── docs/                    # Documentation
│   ├── README.md
│   ├── getting_started.md
│   ├── core_concepts.md
│   ├── api_reference.md
│   ├── examples.md
│   ├── advanced_usage.md
│   ├── troubleshooting.md
│   ├── architecture.md
│   └── contributing.md
├── examples/                # Example scripts
│   └── demo.py
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── setup.py                # Package setup
├── README.md               # Project README
├── LICENSE                 # License file
└── .gitignore             # Git ignore file
```

### Development Tools

1. **Code Formatting**:
   ```bash
   # Format code with black
   black context_manager/ tests/
   
   # Sort imports with isort
   isort context_manager/ tests/
   ```

2. **Linting**:
   ```bash
   # Run flake8
   flake8 context_manager/ tests/
   
   # Run mypy for type checking
   mypy context_manager/
   ```

3. **Pre-commit Hooks**:
   ```bash
   # Install pre-commit
   pip install pre-commit
   pre-commit install
   ```

### Configuration Files

**.flake8**:
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist
```

**pyproject.toml**:
```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

1. **Line Length**: 88 characters (Black default)
2. **Import Order**: Use isort with Black profile
3. **Type Hints**: Use type hints for all public APIs
4. **Docstrings**: Use Google-style docstrings

### Code Style Examples

**Good**:
```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class Turn:
    """Represents a single conversation turn."""
    
    user_input: str
    assistant_response: str
    token_count: int
    timestamp: float
    
    @property
    def text(self) -> str:
        """Get the full text of this turn."""
        return f"User: {self.user_input}\nAssistant: {self.assistant_response}"


class ShortTermMemory:
    """Short-term memory for recent conversation turns."""
    
    def __init__(self, max_tokens: int = 8000, token_counter: Optional[TokenCounter] = None):
        """Initialize short-term memory.
        
        Args:
            max_tokens: Maximum tokens to keep in STM
            token_counter: Token counter instance
        """
        self.max_tokens = max_tokens
        self.token_counter = token_counter or TokenCounter()
        self.turns: deque[Turn] = deque()
        self.current_tokens = 0
    
    def add_turn(self, user_input: str, assistant_response: str) -> None:
        """Add a new turn to short-term memory.
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
        """
        turn_text = f"User: {user_input}\nAssistant: {assistant_response}"
        token_count = self.token_counter.count_tokens(turn_text)
        
        turn = Turn(
            user_input=user_input,
            assistant_response=assistant_response,
            token_count=token_count,
            timestamp=time.time()
        )
        
        self.turns.append(turn)
        self.current_tokens += token_count
        self._evict_if_needed()
```

**Bad**:
```python
# No type hints
def add_turn(self, user_input, assistant_response):
    # No docstring
    turn_text = f"User: {user_input}\nAssistant: {assistant_response}"
    token_count = self.token_counter.count_tokens(turn_text)
    # Inconsistent formatting
    turn = Turn(user_input=user_input,assistant_response=assistant_response,token_count=token_count,timestamp=time.time())
    self.turns.append(turn)
    self.current_tokens += token_count
    self._evict_if_needed()
```

### Naming Conventions

1. **Classes**: PascalCase (e.g., `ShortTermMemory`)
2. **Functions/Methods**: snake_case (e.g., `add_turn`)
3. **Variables**: snake_case (e.g., `max_tokens`)
4. **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_MAX_TOKENS`)
5. **Private Methods**: Leading underscore (e.g., `_evict_if_needed`)

### Error Handling

**Good**:
```python
def search(self, query: str, k: int = 7) -> List[Tuple[MemoryEntry, float]]:
    """Search for similar memories.
    
    Args:
        query: Query text
        k: Number of results to return
        
    Returns:
        List of (memory_entry, similarity_score) tuples
        
    Raises:
        ValueError: If query is empty
        RuntimeError: If embedding model is not loaded
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not self.entries:
        return []
    
    try:
        query_embedding = self.embedding_provider.embed_single(query)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        
        similarities = 1 - distances[0] / np.max(distances[0])
        results = []
        
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.entries):
                entry = self.entries[idx]
                results.append((entry, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise RuntimeError(f"Search operation failed: {e}")
```

**Bad**:
```python
def search(self, query, k=7):
    # No error handling
    query_embedding = self.embedding_provider.embed_single(query)
    distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
    # No validation or error handling
    return [(self.entries[idx], 1 - distances[0][i] / np.max(distances[0])) 
            for i, idx in enumerate(indices[0])]
```

## Testing

### Test Structure

```
tests/
├── __init__.py
├── test_context_manager.py      # Main ContextManager tests
├── test_memory.py              # Memory component tests
├── test_llm.py                 # LLM adapter tests
├── test_utils.py               # Utility tests
├── test_integration.py         # Integration tests
└── conftest.py                 # Pytest configuration
```

### Writing Tests

**Example Test**:
```python
import pytest
from unittest.mock import Mock, patch
import numpy as np

from context_manager.core import ContextManager, Config
from context_manager.memory.short_term import ShortTermMemory, Turn
from context_manager.memory.long_term import LongTermMemory
from context_manager.utils.token_counter import TokenCounter


class TestShortTermMemory:
    """Test cases for ShortTermMemory."""
    
    def test_add_turn(self):
        """Test adding a turn to STM."""
        stm = ShortTermMemory(max_tokens=1000)
        
        stm.add_turn("Hello!", "Hi there!")
        
        assert len(stm.turns) == 1
        assert stm.turns[0].user_input == "Hello!"
        assert stm.turns[0].assistant_response == "Hi there!"
        assert stm.current_tokens > 0
    
    def test_eviction(self):
        """Test automatic eviction when capacity exceeded."""
        stm = ShortTermMemory(max_tokens=100)
        
        # Add turns until capacity exceeded
        for i in range(10):
            stm.add_turn(f"Input {i}", f"Response {i}")
        
        # Should have evicted some turns
        assert stm.current_tokens <= stm.max_tokens
    
    def test_get_recent_turns(self):
        """Test retrieving recent turns."""
        stm = ShortTermMemory(max_tokens=1000)
        
        # Add some turns
        for i in range(5):
            stm.add_turn(f"Input {i}", f"Response {i}")
        
        recent = stm.get_recent_turns(3)
        assert len(recent) == 3
        assert recent[0].user_input == "Input 2"  # Most recent first


class TestLongTermMemory:
    """Test cases for LongTermMemory."""
    
    @patch('context_manager.llm.embeddings.EmbeddingProvider')
    def test_add_memory(self, mock_embedding_provider):
        """Test adding memory to LTM."""
        # Mock embedding provider
        mock_provider = Mock()
        mock_provider.embed_single.return_value = np.array([0.1] * 384)
        mock_embedding_provider.return_value = mock_provider
        
        ltm = LongTermMemory(dimension=384)
        
        memory_id = ltm.add_memory("Test memory", {"type": "test"})
        
        assert memory_id.startswith("mem_")
        assert len(ltm.entries) == 1
        assert ltm.entries[0].text == "Test memory"
    
    @patch('context_manager.llm.embeddings.EmbeddingProvider')
    def test_search(self, mock_embedding_provider):
        """Test searching LTM."""
        # Mock embedding provider
        mock_provider = Mock()
        mock_provider.embed_single.return_value = np.array([0.1] * 384)
        mock_embedding_provider.return_value = mock_provider
        
        ltm = LongTermMemory(dimension=384)
        
        # Add some memories
        ltm.add_memory("Python programming", {"type": "technical"})
        ltm.add_memory("Machine learning", {"type": "technical"})
        ltm.add_memory("Weather today", {"type": "general"})
        
        # Search
        results = ltm.search("programming", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)


class TestContextManager:
    """Test cases for ContextManager."""
    
    @patch('context_manager.llm.adapters.create_llm_adapter')
    @patch('context_manager.llm.embeddings.EmbeddingProvider')
    def test_initialization(self, mock_embedding, mock_llm):
        """Test ContextManager initialization."""
        # Mock dependencies
        mock_embedding.return_value = Mock()
        mock_llm.return_value = Mock()
        
        cm = ContextManager()
        
        assert cm.config is not None
        assert cm.short_term_memory is not None
        assert cm.long_term_memory is not None
        assert cm.summarizer is not None
    
    def test_observe(self):
        """Test observing interactions."""
        cm = ContextManager()
        
        cm.observe("Hello!", "Hi there!")
        
        assert len(cm.short_term_memory.turns) == 1
        assert cm.short_term_memory.turns[0].user_input == "Hello!"
    
    def test_build_context(self):
        """Test context building."""
        cm = ContextManager()
        
        # Add some conversation history
        cm.observe("Hello!", "Hi there!")
        cm.observe("How are you?", "I'm doing well!")
        
        context = cm.build_context("What did we discuss?")
        
        assert "Hello!" in context
        assert "How are you?" in context
        assert "What did we discuss?" in context
    
    def test_query_memory(self):
        """Test memory querying."""
        cm = ContextManager()
        
        # Add some memories
        cm.add_memory("Python is a programming language", {"type": "fact"})
        cm.add_memory("Machine learning uses algorithms", {"type": "fact"})
        
        results = cm.query_memory("programming", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, tuple) for result in results)


class TestTokenCounter:
    """Test cases for TokenCounter."""
    
    def test_count_tokens(self):
        """Test token counting."""
        counter = TokenCounter("gpt-4")
        
        text = "Hello, world!"
        token_count = counter.count_tokens(text)
        
        assert token_count > 0
        assert isinstance(token_count, int)
    
    def test_truncate_to_tokens(self):
        """Test token truncation."""
        counter = TokenCounter("gpt-4")
        
        long_text = "This is a very long text " * 100
        truncated = counter.truncate_to_tokens(long_text, 10)
        
        assert counter.count_tokens(truncated) <= 10


if __name__ == "__main__":
    pytest.main([__file__])
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=context_manager

# Run specific test file
pytest tests/test_context_manager.py

# Run specific test
pytest tests/test_context_manager.py::TestContextManager::test_observe

# Run with verbose output
pytest -v

# Run with parallel execution
pytest -n auto
```

### Test Configuration

**conftest.py**:
```python
import pytest
from context_manager import ContextManager, Config


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return Config(
        memory=MemoryConfig(
            stm_capacity=1000,
            chunk_size=500,
            recent_k=3,
            ltm_hits_k=5,
            prompt_token_budget=2000,
        ),
        debug=True,
        log_level="DEBUG"
    )


@pytest.fixture
def context_manager(sample_config):
    """Provide a ContextManager instance for testing."""
    return ContextManager(sample_config)


@pytest.fixture
def sample_conversation():
    """Provide sample conversation data."""
    return [
        ("Hello!", "Hi there! How can I help you?"),
        ("I need help with Python", "Python is a great language! What specific help do you need?"),
        ("How do I install packages?", "You can use pip: pip install package_name"),
        ("What about virtual environments?", "Use venv: python -m venv myenv"),
    ]
```

## Documentation

### Documentation Standards

1. **Docstrings**: Use Google-style docstrings for all public APIs
2. **Type Hints**: Include type hints in docstrings
3. **Examples**: Provide usage examples
4. **Cross-references**: Link to related functions/classes

### Writing Documentation

**Good Docstring**:
```python
def build_context(self, user_input: str) -> str:
    """Build optimized context for LLM call.
    
    This method combines recent conversation turns from short-term memory
    with relevant memories from long-term memory to create an optimized
    context that fits within the token budget.
    
    Args:
        user_input: Current user input to include in context
        
    Returns:
        Optimized context string ready for LLM consumption
        
    Raises:
        ValueError: If user_input is empty or None
        RuntimeError: If memory components are not properly initialized
        
    Example:
        >>> cm = ContextManager()
        >>> cm.observe("Hello!", "Hi there!")
        >>> context = cm.build_context("What did we discuss?")
        >>> print(len(context))
        150
        
    See Also:
        :meth:`observe`: Add conversation turns to memory
        :meth:`query_memory`: Search long-term memory
    """
```

**Documentation Files**:
- **README.md**: Project overview and quick start
- **getting_started.md**: Installation and basic usage
- **core_concepts.md**: Architecture and concepts
- **api_reference.md**: Complete API documentation
- **examples.md**: Usage examples and patterns
- **advanced_usage.md**: Advanced features and customization
- **troubleshooting.md**: Common issues and solutions
- **architecture.md**: System design and performance
- **contributing.md**: This file

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation
   - Run tests and linting

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat: add support for Anthropic Claude models
fix(memory): resolve token counting issue in STM
docs: update API reference with new methods
test: add integration tests for summarization
```

### Submitting a PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**:
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Test addition
   
   ## Testing
   - [ ] Added tests for new functionality
   - [ ] All tests pass
   - [ ] Updated existing tests if needed
   
   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated README if needed
   - [ ] Updated API docs if needed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Added type hints
   - [ ] Added error handling
   ```

## Code Review Guidelines

### For Contributors

1. **Respond to feedback**: Address all review comments
2. **Keep commits focused**: One logical change per commit
3. **Test thoroughly**: Ensure all tests pass
4. **Update documentation**: Keep docs in sync with code

### For Reviewers

1. **Be constructive**: Provide helpful, specific feedback
2. **Check functionality**: Ensure the code works as intended
3. **Review tests**: Verify adequate test coverage
4. **Check documentation**: Ensure docs are updated

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Type hints are present
- [ ] Error handling is adequate
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No security issues
- [ ] Performance is acceptable
- [ ] Backward compatibility maintained

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version**:
   ```bash
   # Update version in setup.py and __init__.py
   git add .
   git commit -m "chore: bump version to 1.2.0"
   ```

2. **Create release branch**:
   ```bash
   git checkout -b release/1.2.0
   git push origin release/1.2.0
   ```

3. **Run release checks**:
   ```bash
   # Run all tests
   pytest
   
   # Run linting
   flake8 context_manager/ tests/
   
   # Check documentation
   python -m pydoc context_manager
   ```

4. **Create GitHub release**:
   - Go to GitHub releases
   - Create new release
   - Tag with version (e.g., v1.2.0)
   - Write release notes

5. **Publish to PyPI**:
   ```bash
   # Build distribution
   python setup.py sdist bdist_wheel
   
   # Upload to PyPI
   twine upload dist/*
   ```

### Release Notes Template

```markdown
# ContextManager v1.2.0

## 🚀 New Features
- Added support for Anthropic Claude models
- Implemented async context building
- Added memory persistence across sessions

## 🐛 Bug Fixes
- Fixed token counting for non-ASCII characters
- Resolved memory leak in long-term memory
- Fixed embedding model download issues

## 📚 Documentation
- Updated API reference with new methods
- Added troubleshooting guide
- Improved getting started documentation

## 🔧 Improvements
- Enhanced error handling and logging
- Improved performance for large memory sets
- Better configuration validation

## 📦 Installation
```bash
pip install context-manager==1.2.0
```

## 🔗 Links
- [Documentation](https://contextmanager.readthedocs.io/)
- [GitHub Repository](https://github.com/contextmanager/context-manager)
- [Issue Tracker](https://github.com/contextmanager/context-manager/issues)
```

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: support@contextmanager.ai
- **Discord**: [ContextManager Community](https://discord.gg/contextmanager)

### Issue Templates

**Bug Report**:
```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 12.0]
- Python: [e.g., 3.9.7]
- ContextManager: [e.g., 1.1.0]

## Additional Information
Any other relevant information
```

**Feature Request**:
```markdown
## Feature Description
Brief description of the feature

## Use Case
Why this feature is needed

## Proposed Solution
How you think it should work

## Alternatives Considered
Other approaches you've considered

## Additional Information
Any other relevant information
```

---

Thank you for contributing to ContextManager! Your contributions help make this project better for everyone. 