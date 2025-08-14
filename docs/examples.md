# Examples

This section provides practical examples and use cases for ContextManager, demonstrating how to integrate it with different types of AI agents and applications.

## 📚 Table of Contents

- [Basic Agent Integration](#basic-agent-integration)
- [Tool-Using Agents](#tool-using-agents)
- [Multi-Session Persistence](#multi-session-persistence)
- [Custom Summarization](#custom-summarization)
- [Memory Querying Patterns](#memory-querying-patterns)
- [Performance Optimization](#performance-optimization)
- [Debugging and Monitoring](#debugging-and-monitoring)
- [Advanced Patterns](#advanced-patterns)

## Basic Agent Integration

### Simple Chat Agent

The most basic integration pattern for a chat agent.

```python
from context_manager import ContextManager
import openai

class SimpleChatAgent:
    def __init__(self, api_key: str):
        self.cm = ContextManager()
        openai.api_key = api_key
    
    def chat(self, user_input: str) -> str:
        # Build context with memory
        context = self.cm.build_context(user_input)
        
        # Call LLM
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context}
            ],
            max_tokens=500
        )
        
        assistant_response = response.choices[0].message.content
        
        # Observe the interaction
        self.cm.observe(user_input, assistant_response)
        
        return assistant_response

# Usage
agent = SimpleChatAgent("your-openai-key")

conversation = [
    "Hello! I'm planning a trip to Japan.",
    "I want to visit Tokyo and Kyoto. What should I know?",
    "What's the best time to visit?",
    "How much should I budget for 10 days?",
    "Can you remind me what we discussed about my Japan trip?"
]

for user_input in conversation:
    response = agent.chat(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {response}")
    print("-" * 50)
```

### Agent with Custom Configuration

Example with custom memory and LLM settings.

```python
from context_manager import Config, ContextManager, MemoryConfig, LLMConfig
from context_manager.llm.adapters import create_llm_adapter

class CustomAgent:
    def __init__(self):
        # Custom configuration
        config = Config(
            memory=MemoryConfig(
                stm_capacity=12000,        # Larger STM
                chunk_size=3000,           # Larger chunks
                recent_k=8,                # More recent turns
                ltm_hits_k=10,            # More LTM results
                prompt_token_budget=16000, # Larger context
            ),
            llm=LLMConfig(
                provider="anthropic",
                model="claude-3-sonnet-20240229",
                api_key="your-anthropic-key",
                max_tokens=1500,
                temperature=0.5,
            ),
            debug=True,
            log_level="DEBUG"
        )
        
        self.cm = ContextManager(config)
        self.llm = create_llm_adapter("anthropic", api_key="your-anthropic-key")
    
    def respond(self, user_input: str) -> str:
        context = self.cm.build_context(user_input)
        
        response = self.llm.generate_sync(
            f"User: {context}\n\nAssistant:",
            max_tokens=1500,
            temperature=0.5
        )
        
        self.cm.observe(user_input, response)
        return response

# Usage
agent = CustomAgent()
response = agent.respond("Tell me about machine learning")
print(response)
```

## Tool-Using Agents

### Agent with External Tools

Example of an agent that uses tools but has no built-in memory.

```python
from context_manager import ContextManager
from context_manager.llm.adapters import create_llm_adapter
import requests
import json

class ToolAgent:
    def __init__(self):
        self.cm = ContextManager()
        self.llm = create_llm_adapter("openai", api_key="your-key")
        
        # Define available tools
        self.tools = {
            "search_web": self._search_web,
            "get_weather": self._get_weather,
            "calculate": self._calculate,
            "get_time": self._get_time,
            "search_wikipedia": self._search_wikipedia
        }
    
    def _search_web(self, query: str) -> str:
        """Mock web search tool."""
        return f"Search results for '{query}': Found 5 relevant pages about {query}."
    
    def _get_weather(self, location: str) -> str:
        """Mock weather tool."""
        return f"Weather in {location}: 72°F, partly cloudy with 20% chance of rain."
    
    def _calculate(self, expression: str) -> str:
        """Calculator tool."""
        try:
            result = eval(expression)
            return f"Calculation result: {expression} = {result}"
        except:
            return f"Error calculating: {expression}"
    
    def _get_time(self) -> str:
        """Time tool."""
        import datetime
        return f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _search_wikipedia(self, topic: str) -> str:
        """Mock Wikipedia search."""
        return f"Wikipedia article on '{topic}': {topic} is a fascinating subject with rich history."
    
    def respond(self, user_input: str) -> str:
        # Build context with memory
        context = self.cm.build_context(user_input)
        
        # Create tool prompt
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        ])
        
        prompt = f"""You are a helpful AI assistant with access to the following tools:

{tool_descriptions}

When the user asks for information that requires these tools, use them appropriately.
Always respond naturally and conversationally.

User: {context}

Assistant:"""
        
        # Generate response
        response = self.llm.generate_sync(prompt, max_tokens=1000)
        
        # Observe interaction
        self.cm.observe(user_input, response)
        
        return response

# Usage
agent = ToolAgent()

conversation = [
    "What's the weather like in Tokyo?",
    "Calculate 15 * 23 for me.",
    "What time is it right now?",
    "Search for information about Python programming.",
    "Can you remind me what we discussed about Tokyo?"
]

for user_input in conversation:
    response = agent.respond(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {response}")
    print("-" * 50)
```

### Agent with Dynamic Tool Selection

Advanced example with dynamic tool selection based on context.

```python
from context_manager import ContextManager
from context_manager.llm.adapters import create_llm_adapter
import re

class DynamicToolAgent:
    def __init__(self):
        self.cm = ContextManager()
        self.llm = create_llm_adapter("openai", api_key="your-key")
        
        self.tools = {
            "calculator": self._calculator,
            "weather": self._weather,
            "search": self._search,
            "time": self._time,
            "converter": self._converter
        }
    
    def _calculator(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"Result: {expression} = {result}"
        except:
            return f"Error: Invalid expression '{expression}'"
    
    def _weather(self, location: str) -> str:
        return f"Weather in {location}: 75°F, sunny with light breeze."
    
    def _search(self, query: str) -> str:
        return f"Search results for '{query}': Found relevant information about {query}."
    
    def _time(self) -> str:
        import datetime
        return f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _converter(self, value: str, from_unit: str, to_unit: str) -> str:
        # Mock conversion
        return f"Converted {value} {from_unit} to {value * 2.54} {to_unit}"
    
    def _extract_tool_calls(self, response: str) -> list:
        """Extract tool calls from response."""
        tool_calls = []
        # Simple regex-based extraction (in practice, use more sophisticated parsing)
        patterns = [
            r"calculator\(([^)]+)\)",
            r"weather\(([^)]+)\)",
            r"search\(([^)]+)\)",
            r"time\(\)",
            r"converter\(([^)]+)\)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                tool_calls.append((pattern.split('(')[0], match))
        
        return tool_calls
    
    def respond(self, user_input: str) -> str:
        # Build context
        context = self.cm.build_context(user_input)
        
        # Generate response with potential tool calls
        prompt = f"""You are an AI assistant with access to tools. When you need to use a tool, 
format your response as: tool_name(parameters). Available tools:
- calculator(expression): Calculate mathematical expressions
- weather(location): Get weather for location
- search(query): Search for information
- time(): Get current time
- converter(value, from_unit, to_unit): Convert units

User: {context}

Assistant:"""
        
        response = self.llm.generate_sync(prompt, max_tokens=1000)
        
        # Extract and execute tool calls
        tool_calls = self._extract_tool_calls(response)
        tool_results = []
        
        for tool_name, params in tool_calls:
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](params)
                    tool_results.append(f"{tool_name}({params}) = {result}")
                except Exception as e:
                    tool_results.append(f"Error in {tool_name}({params}): {str(e)}")
        
        # Combine response with tool results
        if tool_results:
            final_response = response + "\n\nTool Results:\n" + "\n".join(tool_results)
        else:
            final_response = response
        
        # Observe interaction
        self.cm.observe(user_input, final_response)
        
        return final_response

# Usage
agent = DynamicToolAgent()
response = agent.respond("What's 25 * 4 and what's the weather in Paris?")
print(response)
```

## Multi-Session Persistence

### Persistent Memory Across Sessions

Example showing how to maintain memory across multiple sessions.

```python
import pickle
import os
from context_manager import ContextManager

class PersistentAgent:
    def __init__(self, session_id: str, memory_file: str = None):
        self.session_id = session_id
        self.memory_file = memory_file or f"memory_{session_id}.pkl"
        
        # Try to load existing memory
        self.cm = self._load_memory() or ContextManager()
    
    def _load_memory(self) -> ContextManager:
        """Load memory from file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load memory: {e}")
        return None
    
    def _save_memory(self):
        """Save memory to file."""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.cm, f)
        except Exception as e:
            print(f"Failed to save memory: {e}")
    
    def chat(self, user_input: str) -> str:
        # Build context
        context = self.cm.build_context(user_input)
        
        # Call LLM (simplified)
        response = f"Response to: {user_input}"
        
        # Observe and save
        self.cm.observe(user_input, response)
        self._save_memory()
        
        return response
    
    def get_memory_stats(self):
        """Get memory statistics."""
        stats = self.cm.get_stats()
        return {
            "session_id": self.session_id,
            "stm_turns": stats['short_term_memory']['num_turns'],
            "ltm_entries": stats['long_term_memory']['num_entries'],
            "memory_file": self.memory_file
        }

# Usage across sessions
session_id = "user_123"

# Session 1
agent1 = PersistentAgent(session_id)
agent1.chat("Hello! I'm planning a trip to Japan.")
agent1.chat("I want to visit Tokyo and Kyoto.")
print("Session 1 stats:", agent1.get_memory_stats())

# Session 2 (later)
agent2 = PersistentAgent(session_id)
agent2.chat("Can you remind me what we discussed about my Japan trip?")
print("Session 2 stats:", agent2.get_memory_stats())
```

### Multi-User Session Management

Advanced example with multiple user sessions.

```python
from context_manager import ContextManager
import json
import os
from datetime import datetime

class MultiUserAgent:
    def __init__(self, data_dir: str = "user_sessions"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.active_sessions = {}
    
    def get_session(self, user_id: str) -> ContextManager:
        """Get or create session for user."""
        if user_id not in self.active_sessions:
            session_file = os.path.join(self.data_dir, f"{user_id}.json")
            self.active_sessions[user_id] = self._load_session(session_file)
        
        return self.active_sessions[user_id]
    
    def _load_session(self, session_file: str) -> ContextManager:
        """Load session from file."""
        if os.path.exists(session_file):
            try:
                # Load session data and reconstruct ContextManager
                # (In practice, you'd implement proper serialization)
                return ContextManager()
            except Exception as e:
                print(f"Failed to load session: {e}")
        
        return ContextManager()
    
    def _save_session(self, user_id: str):
        """Save session to file."""
        session_file = os.path.join(self.data_dir, f"{user_id}.json")
        cm = self.active_sessions[user_id]
        
        # Save session data (simplified)
        session_data = {
            "user_id": user_id,
            "last_updated": datetime.now().isoformat(),
            "stats": cm.get_stats()
        }
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save session: {e}")
    
    def chat(self, user_id: str, user_input: str) -> str:
        """Chat with a specific user."""
        cm = self.get_session(user_id)
        
        # Build context
        context = cm.build_context(user_input)
        
        # Call LLM (simplified)
        response = f"Response to user {user_id}: {user_input}"
        
        # Observe and save
        cm.observe(user_input, response)
        self._save_session(user_id)
        
        return response
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get statistics for a specific user."""
        cm = self.get_session(user_id)
        stats = cm.get_stats()
        return {
            "user_id": user_id,
            "stm_turns": stats['short_term_memory']['num_turns'],
            "ltm_entries": stats['long_term_memory']['num_entries'],
            "last_updated": datetime.now().isoformat()
        }

# Usage
agent = MultiUserAgent()

# User 1
agent.chat("user_1", "Hello! I'm planning a trip to Japan.")
agent.chat("user_1", "I want to visit Tokyo and Kyoto.")

# User 2
agent.chat("user_2", "Hi! I need help with Python programming.")
agent.chat("user_2", "What are the best practices for error handling?")

# Check stats
print("User 1 stats:", agent.get_user_stats("user_1"))
print("User 2 stats:", agent.get_user_stats("user_2"))
```

## Custom Summarization

### Custom Summarization Prompts

Example with custom summarization prompts for specific use cases.

```python
from context_manager.memory.summarizer import HierarchicalSummarizer
from context_manager.llm.adapters import create_llm_adapter

class CustomSummarizer(HierarchicalSummarizer):
    def __init__(self, llm_adapter=None, compression_ratio=0.3):
        super().__init__(llm_adapter, compression_ratio)
        
        # Custom prompts for different use cases
        self.custom_prompts = {
            "technical": """
Summarize the following technical conversation, focusing on:
1. Technical concepts discussed
2. Code examples and solutions
3. Best practices mentioned
4. Tools and technologies referenced
5. Problem-solving approaches

Conversation:
{text}

Technical Summary:""",
            
            "planning": """
Summarize the following planning conversation, focusing on:
1. Goals and objectives
2. Timeline and deadlines
3. Resources and requirements
4. Action items and next steps
5. Risks and considerations

Conversation:
{text}

Planning Summary:""",
            
            "creative": """
Summarize the following creative conversation, focusing on:
1. Ideas and concepts generated
2. Creative directions explored
3. Inspiration sources mentioned
4. Feedback and iterations
5. Final decisions and outcomes

Conversation:
{text}

Creative Summary:"""
        }
    
    def summarize_chunk(self, turns, conversation_type="general"):
        """Summarize with custom prompts based on conversation type."""
        if conversation_type in self.custom_prompts:
            combined_text = "\n\n".join([turn.text for turn in turns])
            prompt = self.custom_prompts[conversation_type].format(text=combined_text)
            
            try:
                summary = self.llm_adapter.generate_sync(
                    prompt,
                    max_tokens=500,
                    temperature=0.3
                )
                return summary.strip()
            except Exception as e:
                print(f"Custom summarization failed: {e}")
                return super().summarize_chunk(turns)
        
        return super().summarize_chunk(turns)

# Usage
llm_adapter = create_llm_adapter("openai", api_key="your-key")
custom_summarizer = CustomSummarizer(llm_adapter)

# Technical conversation
technical_turns = [
    # Your technical conversation turns
]
technical_summary = custom_summarizer.summarize_chunk(technical_turns, "technical")

# Planning conversation
planning_turns = [
    # Your planning conversation turns
]
planning_summary = custom_summarizer.summarize_chunk(planning_turns, "planning")
```

## Memory Querying Patterns

### Semantic Search Examples

Different patterns for querying memory effectively.

```python
from context_manager import ContextManager

class MemoryQueryAgent:
    def __init__(self):
        self.cm = ContextManager()
    
    def find_relevant_context(self, query: str, k: int = 5) -> list:
        """Find relevant context for a query."""
        results = self.cm.query_memory(query, k=k)
        return [(text, score) for text, score in results if score > 0.5]
    
    def search_by_topic(self, topic: str) -> list:
        """Search for memories related to a specific topic."""
        return self.find_relevant_context(f"discussions about {topic}")
    
    def search_by_time_period(self, period: str) -> list:
        """Search for memories from a specific time period."""
        return self.find_relevant_context(f"conversations from {period}")
    
    def search_decisions(self) -> list:
        """Search for decision-related memories."""
        return self.find_relevant_context("decisions made choices selected")
    
    def search_preferences(self) -> list:
        """Search for user preference memories."""
        return self.find_relevant_context("preferences likes dislikes prefer")
    
    def search_action_items(self) -> list:
        """Search for action items and tasks."""
        return self.find_relevant_context("action items tasks todo next steps")

# Usage
agent = MemoryQueryAgent()

# Add some conversation history
conversation = [
    ("I prefer Python over JavaScript", "That's a great choice! Python is very versatile."),
    ("I decided to use React for the frontend", "React is excellent for building user interfaces."),
    ("My budget is $5000 for this project", "That's a reasonable budget for this scope."),
    ("I need to finish this by next Friday", "That gives you about a week to complete it."),
    ("I like using Docker for deployment", "Docker is great for containerization.")
]

for user_input, response in conversation:
    agent.cm.observe(user_input, response)

# Search patterns
print("Python preferences:", agent.search_by_topic("Python"))
print("Decisions made:", agent.search_decisions())
print("Budget info:", agent.find_relevant_context("budget"))
print("Action items:", agent.search_action_items())
```

### Advanced Memory Analytics

Example with memory analytics and insights.

```python
from context_manager import ContextManager
from collections import Counter
import re

class MemoryAnalytics:
    def __init__(self, cm: ContextManager):
        self.cm = cm
    
    def get_memory_insights(self) -> dict:
        """Get insights about stored memories."""
        results = self.cm.query_memory("", k=100)  # Get all memories
        
        # Analyze memory content
        all_text = " ".join([text for text, _ in results])
        
        # Extract topics
        topics = self._extract_topics(all_text)
        
        # Extract sentiment patterns
        sentiment = self._analyze_sentiment(all_text)
        
        # Extract key entities
        entities = self._extract_entities(all_text)
        
        return {
            "total_memories": len(results),
            "topics": topics,
            "sentiment": sentiment,
            "entities": entities,
            "memory_distribution": self._analyze_memory_distribution(results)
        }
    
    def _extract_topics(self, text: str) -> dict:
        """Extract common topics from memory."""
        # Simple keyword extraction (in practice, use NLP libraries)
        keywords = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(keywords)
        
        # Filter common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        topics = {word: count for word, count in word_freq.items() 
                 if word not in common_words and len(word) > 3}
        
        return dict(topics.most_common(10))
    
    def _analyze_sentiment(self, text: str) -> dict:
        """Simple sentiment analysis."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'prefer']
        negative_words = ['bad', 'terrible', 'hate', 'dislike', 'problem', 'issue', 'difficult']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        return {
            "positive": positive_count,
            "negative": negative_count,
            "ratio": positive_count / (positive_count + negative_count + 1)
        }
    
    def _extract_entities(self, text: str) -> dict:
        """Extract named entities (simplified)."""
        # Simple entity extraction (in practice, use NER)
        entities = {
            "technologies": re.findall(r'\b(Python|JavaScript|React|Docker|AWS|Azure)\b', text, re.IGNORECASE),
            "numbers": re.findall(r'\$\d+', text),
            "dates": re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text),
        }
        
        return {k: list(set(v)) for k, v in entities.items()}
    
    def _analyze_memory_distribution(self, results: list) -> dict:
        """Analyze distribution of memory scores."""
        scores = [score for _, score in results]
        
        return {
            "average_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "high_confidence": len([s for s in scores if s > 0.8]),
            "medium_confidence": len([s for s in scores if 0.5 <= s <= 0.8]),
            "low_confidence": len([s for s in scores if s < 0.5])
        }

# Usage
cm = ContextManager()
analytics = MemoryAnalytics(cm)

# Add some memories
memories = [
    "I love Python programming and prefer it over JavaScript",
    "React is great for building user interfaces",
    "I have a budget of $5000 for this project",
    "The deadline is 12/15/2024",
    "Docker makes deployment much easier"
]

for memory in memories:
    cm.add_memory(memory)

# Get insights
insights = analytics.get_memory_insights()
print("Memory Insights:")
print(f"Total memories: {insights['total_memories']}")
print(f"Top topics: {insights['topics']}")
print(f"Sentiment: {insights['sentiment']}")
print(f"Entities: {insights['entities']}")
print(f"Distribution: {insights['memory_distribution']}")
```

## Performance Optimization

### Optimized Configuration for Different Use Cases

```python
from context_manager import Config, MemoryConfig, LLMConfig

class OptimizedAgent:
    def __init__(self, use_case: str = "general"):
        config = self._get_optimized_config(use_case)
        self.cm = ContextManager(config)
    
    def _get_optimized_config(self, use_case: str) -> Config:
        """Get optimized configuration for specific use case."""
        
        if use_case == "fast_response":
            # Optimized for speed
            return Config(
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
                    max_tokens=500,         # Shorter responses
                    temperature=0.3,        # More focused
                )
            )
        
        elif use_case == "high_quality":
            # Optimized for quality
            return Config(
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
                )
            )
        
        elif use_case == "memory_constrained":
            # Optimized for limited memory
            return Config(
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
                    max_tokens=300,
                    temperature=0.5,
                )
            )
        
        else:
            # Default configuration
            return Config()

# Usage
fast_agent = OptimizedAgent("fast_response")
quality_agent = OptimizedAgent("high_quality")
constrained_agent = OptimizedAgent("memory_constrained")

# Compare performance
import time

def benchmark_agent(agent, input_text: str):
    start_time = time.time()
    context = agent.cm.build_context(input_text)
    build_time = time.time() - start_time
    
    stats = agent.cm.get_stats()
    
    return {
        "build_time": build_time,
        "context_length": len(context),
        "stm_turns": stats['short_term_memory']['num_turns'],
        "ltm_entries": stats['long_term_memory']['num_entries']
    }

# Benchmark
test_input = "What did we discuss about Python programming?"
results = {}

for name, agent in [("Fast", fast_agent), ("Quality", quality_agent), ("Constrained", constrained_agent)]:
    results[name] = benchmark_agent(agent, test_input)

for name, result in results.items():
    print(f"{name} Agent:")
    print(f"  Build time: {result['build_time']:.3f}s")
    print(f"  Context length: {result['context_length']} chars")
    print(f"  STM turns: {result['stm_turns']}")
    print(f"  LTM entries: {result['ltm_entries']}")
    print()
```

## Debugging and Monitoring

### Debug Context Building

```python
from context_manager import ContextManager

class DebugAgent:
    def __init__(self):
        self.cm = ContextManager()
    
    def debug_context(self, user_input: str) -> dict:
        """Debug context building process."""
        debug_info = self.cm.debug_context_building(user_input)
        
        print("🔍 Context Building Debug:")
        print(f"  User input: {debug_info['user_input']}")
        print(f"  Recent turns: {debug_info['recent_turns_count']}")
        print(f"  LTM hits: {debug_info['ltm_results_count']}")
        print(f"  Final context length: {debug_info['final_context_length']}")
        print(f"  Final context tokens: {debug_info['final_context_tokens']}")
        print(f"  Context budget: {debug_info['context_budget']}")
        print(f"  Token utilization: {debug_info['final_context_tokens'] / debug_info['context_budget']:.2%}")
        
        if debug_info['recent_texts']:
            print("\n  Recent turns:")
            for i, text in enumerate(debug_info['recent_texts']):
                print(f"    {i+1}. {text[:100]}...")
        
        if debug_info['ltm_results']:
            print("\n  LTM results:")
            for i, (text, score) in enumerate(debug_info['ltm_results']):
                print(f"    {i+1}. Score {score:.2f}: {text[:100]}...")
        
        return debug_info
    
    def monitor_memory_usage(self):
        """Monitor memory usage and performance."""
        stats = self.cm.get_stats()
        
        print("📊 Memory Usage Monitor:")
        print(f"  STM turns: {stats['short_term_memory']['num_turns']}")
        print(f"  STM tokens: {stats['short_term_memory']['current_tokens']}")
        print(f"  STM utilization: {stats['short_term_memory']['utilization']:.2%}")
        print(f"  LTM entries: {stats['long_term_memory']['num_entries']}")
        print(f"  LTM index size: {stats['long_term_memory']['index_size']}")
        
        return stats

# Usage
agent = DebugAgent()

# Add some conversation
agent.cm.observe("Hello!", "Hi there!")
agent.cm.observe("How are you?", "I'm doing well!")
agent.cm.observe("What's the weather?", "It's sunny today!")

# Debug context building
debug_info = agent.debug_context("What did we discuss?")

# Monitor memory
stats = agent.monitor_memory_usage()
```

## Advanced Patterns

### Multi-Agent Shared Memory

```python
from context_manager import ContextManager
import threading
import time

class SharedMemoryManager:
    def __init__(self):
        self.shared_cm = ContextManager()
        self.lock = threading.Lock()
    
    def add_memory(self, agent_id: str, user_input: str, response: str):
        """Add memory with agent identification."""
        with self.lock:
            self.shared_cm.observe(user_input, response)
            # Add agent metadata
            self.shared_cm.add_memory(
                f"Agent {agent_id}: {user_input} -> {response}",
                {"agent_id": agent_id, "timestamp": time.time()}
            )
    
    def query_memory(self, query: str, agent_id: str = None) -> list:
        """Query memory, optionally filtered by agent."""
        with self.lock:
            results = self.shared_cm.query_memory(query)
            
            if agent_id:
                # Filter by agent if specified
                filtered_results = []
                for text, score in results:
                    if f"Agent {agent_id}:" in text:
                        filtered_results.append((text, score))
                return filtered_results
            
            return results

class MultiAgentSystem:
    def __init__(self):
        self.memory_manager = SharedMemoryManager()
        self.agents = {}
    
    def add_agent(self, agent_id: str, cm: ContextManager):
        """Add an agent to the system."""
        self.agents[agent_id] = cm
    
    def agent_chat(self, agent_id: str, user_input: str) -> str:
        """Chat with a specific agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        cm = self.agents[agent_id]
        context = cm.build_context(user_input)
        
        # Generate response (simplified)
        response = f"Agent {agent_id}: {user_input}"
        
        # Update both local and shared memory
        cm.observe(user_input, response)
        self.memory_manager.add_memory(agent_id, user_input, response)
        
        return response
    
    def get_shared_insights(self, query: str) -> list:
        """Get insights from shared memory."""
        return self.memory_manager.query_memory(query)

# Usage
system = MultiAgentSystem()

# Add agents
agent1_cm = ContextManager()
agent2_cm = ContextManager()
system.add_agent("assistant", agent1_cm)
system.add_agent("specialist", agent2_cm)

# Agent interactions
system.agent_chat("assistant", "Hello! How can I help?")
system.agent_chat("specialist", "I need help with Python programming.")
system.agent_chat("assistant", "What specific Python topic do you need help with?")

# Query shared memory
insights = system.get_shared_insights("Python programming")
print("Shared insights:", insights)
```

---

**Next**: [Advanced Usage](./advanced_usage.md) → Advanced patterns and custom implementations 