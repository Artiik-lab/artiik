# Contributing

Thank you for your interest in contributing to ContextManager! This guide will help you get started.

## 🚀 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda

### Clone and Setup

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
```

### Verify Setup

```bash
# Run tests to verify everything works
pytest

# Run linting
black context_manager/
isort context_manager/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=context_manager

# Run specific test file
pytest tests/test_context_manager.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### Writing Tests

Tests should be placed in the `tests/` directory and follow these conventions:

```python
# tests/test_example.py
import pytest
from artiik import ContextManager

def test_basic_functionality():
    """Test basic ContextManager functionality."""
    cm = ContextManager()
    
    # Test observation
    cm.observe("Hello!", "Hi there!")
    
    # Test context building
    context = cm.build_context("What did we discuss?")
    assert "Hello" in context
    
    # Test memory querying
    results = cm.query_memory("greeting")
    assert len(results) > 0
```

### Test Guidelines

- **Descriptive names**: Test function names should clearly describe what they test
- **Isolation**: Each test should be independent and not rely on other tests
- **Coverage**: Aim for high test coverage, especially for core functionality
- **Edge cases**: Test edge cases and error conditions

## 📝 Code Style

### Formatting

We use [Black](https://black.readthedocs.io/) for code formatting and [isort](https://pycqa.github.io/isort/) for import sorting.

```bash
# Format code
black context_manager/

# Sort imports
isort context_manager/

# Check formatting without making changes
black --check context_manager/
isort --check-only context_manager/
```

### Linting

We use [mypy](https://mypy.readthedocs.io/) for type checking.

```bash
# Run type checking
mypy context_manager/
```

### Code Style Guidelines

- **Type hints**: Use type hints for all function parameters and return values
- **Docstrings**: Include docstrings for all public functions and classes
- **Naming**: Use descriptive variable and function names
- **Comments**: Add comments for complex logic

Example:

```python
from typing import List, Optional, Dict, Any

def build_context(self, user_input: str) -> str:
    """
    Build optimized context for LLM input.
    
    Args:
        user_input: Current user input text
        
    Returns:
        Optimized context string ready for LLM
        
    Raises:
        ValueError: If user_input is empty
    """
    if not user_input.strip():
        raise ValueError("user_input cannot be empty")
    
    # Get recent turns from STM
    recent_turns = self.short_term_memory.get_recent_turns(
        self.config.memory.recent_k
    )
    
    # Search LTM for relevant memories
    ltm_results = self.long_term_memory.search(
        user_input, 
        k=self.config.memory.ltm_hits_k
    )
    
    # Assemble and return context
    return self._assemble_context(recent_turns, ltm_results, user_input)
```

## 🔄 Development Workflow

### 1. Create a Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### 2. Make Changes

- Write your code following the style guidelines
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run linting
black --check context_manager/
isort --check-only context_manager/
mypy context_manager/
```

### 4. Commit Your Changes

```bash
# Add your changes
git add .

# Commit with descriptive message
git commit -m "feat: add custom memory implementation

- Add CustomShortTermMemory class
- Add priority-based turn selection
- Add comprehensive tests
- Update documentation"
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📋 Pull Request Guidelines

### Before Submitting

1. **Tests pass**: Ensure all tests pass
2. **Code style**: Run formatting and linting
3. **Documentation**: Update docs if adding new features
4. **Type hints**: Add type hints for new functions
5. **Coverage**: Ensure good test coverage

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] Documentation updated
- [ ] No breaking changes (or breaking changes documented)
```

## 🐛 Bug Reports

### Before Reporting

1. **Check existing issues**: Search for similar issues
2. **Reproduce**: Ensure you can reproduce the issue
3. **Minimal example**: Create a minimal example that reproduces the issue

### Bug Report Template

```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- Python version: 3.9.0
- ContextManager version: 0.1.0
- OS: macOS 12.0

## Additional Context
Any other context about the problem
```

## 💡 Feature Requests

### Before Requesting

1. **Check existing features**: Ensure the feature doesn't already exist
2. **Use case**: Clearly describe the use case
3. **Implementation**: Consider how it might be implemented

### Feature Request Template

```markdown
## Description
Clear description of the feature

## Use Case
Why this feature is needed

## Proposed Implementation
How you think it could be implemented

## Alternatives Considered
Other approaches you considered

## Additional Context
Any other relevant information
```

## 🏗️ Architecture Guidelines

### Adding New Components

When adding new components, follow these guidelines:

1. **Separation of concerns**: Each component should have a single responsibility
2. **Interface consistency**: Follow existing patterns and interfaces
3. **Configuration**: Make components configurable through the Config system
4. **Error handling**: Include proper error handling and validation

### Example Component Structure

```python
# context_manager/new_component/__init__.py
from .component import NewComponent

__all__ = ["NewComponent"]

# context_manager/new_component/component.py
from typing import Optional, Dict, Any
from pydantic import BaseModel

class NewComponentConfig(BaseModel):
    """Configuration for NewComponent."""
    enabled: bool = True
    max_items: int = 100

class NewComponent:
    """New component following ContextManager patterns."""
    
    def __init__(self, config: Optional[NewComponentConfig] = None):
        self.config = config or NewComponentConfig()
    
    def process(self, data: str) -> Dict[str, Any]:
        """Process data and return results."""
        # Implementation here
        pass
```

## 📚 Documentation

### Updating Documentation

When adding new features, update the relevant documentation:

1. **API Reference**: Add new classes and methods to `docs/api_reference.md`
2. **Examples**: Add examples to `docs/examples.md`
3. **Core Concepts**: Update `docs/core_concepts.md` if adding new concepts
4. **README**: Update main README if adding major features

### Documentation Style

- Use clear, concise language
- Include code examples
- Follow existing formatting patterns
- Keep documentation up to date with code changes

## 🚀 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version**: Update version in `setup.py` and `__init__.py`
2. **Update changelog**: Add release notes
3. **Run tests**: Ensure all tests pass
4. **Update documentation**: Ensure docs are current
5. **Create release**: Create GitHub release with tag

## 🤝 Getting Help

### Questions and Discussion

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: boualamhamza@outlook.fr for direct contact

### Community Guidelines

- Be respectful and inclusive
- Help others learn and contribute
- Provide constructive feedback
- Follow the project's code of conduct

---

**Ready to contribute?** Start by setting up your development environment and running the tests! 