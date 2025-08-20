# Advanced Usage

This guide covers advanced patterns and custom implementations for ContextManager.

## 🔧 Custom Memory Implementations

### Custom Short-Term Memory

```python
from context_manager.memory.short_term import ShortTermMemory
from collections import deque
import time

class CustomShortTermMemory(ShortTermMemory):
    def __init__(self, max_tokens: int = 8000, token_counter=None):
        super().__init__(max_tokens, token_counter)
        self.priority_scores = {}  # Track importance of turns
    
    def add_turn(self, user_input: str, assistant_response: str, priority: float = 1.0):
        """Add turn with priority score."""
        turn_id = f"turn_{int(time.time() * 1000)}"
        self.priority_scores[turn_id] = priority
        
        super().add_turn(user_input, assistant_response)
    
    def get_high_priority_turns(self, k: int = 5):
        """Get turns with highest priority scores."""
        if not self.turns:
            return []
        
        # Sort by priority score
        sorted_turns = sorted(
            self.turns, 
            key=lambda turn: self.priority_scores.get(turn.id, 0),
            reverse=True
        )
        
        return sorted_turns[:k]
```

### Custom Long-Term Memory

```python
from context_manager.memory.long_term import LongTermMemory
import numpy as np

class CustomLongTermMemory(LongTermMemory):
    def __init__(self, dimension: int = 384, embedding_provider=None, config=None):
        super().__init__(dimension, embedding_provider, config)
        self.category_index = {}  # Index by categories
    
    def add_memory(self, text: str, metadata: dict = None):
        """Add memory with category indexing."""
        memory_id = super().add_memory(text, metadata)
        
        # Index by category if provided
        if metadata and 'category' in metadata:
            category = metadata['category']
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(memory_id)
        
        return memory_id
    
    def search_by_category(self, category: str, query: str, k: int = 5):
        """Search within a specific category."""
        if category not in self.category_index:
            return []
        
        # Get memory IDs for category
        category_memory_ids = self.category_index[category]
        
        # Filter entries by category
        category_entries = [
            entry for entry in self.entries 
            if entry.id in category_memory_ids
        ]
        
        # Perform search within category
        query_embedding = self.embedding_provider.get_embedding(query)
        results = []
        
        for entry in category_entries:
            similarity = self._calculate_similarity(query_embedding, entry.embedding)
            results.append((entry.text, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
```

## 🎯 Multi-Agent Patterns

### Shared Memory Manager

```python
from artiik import ContextManager
import threading
import time

class SharedMemoryManager:
    def __init__(self):
        self.shared_cm = ContextManager()
        self.lock = threading.Lock()
        self.agent_memories = {}  # Individual agent memories
    
    def get_agent_memory(self, agent_id: str) -> ContextManager:
        """Get or create memory for an agent."""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = ContextManager(
                session_id=f"agent_{agent_id}"
            )
        return self.agent_memories[agent_id]
    
    def add_shared_memory(self, agent_id: str, user_input: str, response: str):
        """Add memory to shared space."""
        with self.lock:
            self.shared_cm.observe(user_input, response)
            # Add agent metadata
            self.shared_cm.ingest_text(
                f"Agent {agent_id}: {user_input} -> {response}",
                importance=0.5
            )
    
    def query_shared_memory(self, query: str, agent_id: str = None):
        """Query shared memory, optionally filtered by agent."""
        with self.lock:
            if agent_id:
                # Filter by agent
                agent_query = f"Agent {agent_id} {query}"
                return self.shared_cm.query_memory(agent_query)
            else:
                return self.shared_cm.query_memory(query)

# Usage
memory_manager = SharedMemoryManager()

# Agent interactions
agent1_memory = memory_manager.get_agent_memory("assistant")
agent2_memory = memory_manager.get_agent_memory("specialist")

# Individual agent memories
agent1_memory.observe("Hello!", "Hi there!")
agent2_memory.observe("I need help with Python", "Python is great!")

# Shared memory
memory_manager.add_shared_memory("assistant", "User asked about Python", "I explained Python basics")
memory_manager.add_shared_memory("specialist", "User needs debugging help", "I provided debugging tips")

# Query shared memory
shared_results = memory_manager.query_shared_memory("Python help")
```

### Agent Coordination

```python
from artiik import ContextManager
from typing import List, Dict

class CoordinatedAgentSystem:
    def __init__(self):
        self.agents = {}
        self.coordinator_memory = ContextManager(session_id="coordinator")
    
    def add_agent(self, agent_id: str, agent_memory: ContextManager):
        """Add an agent to the system."""
        self.agents[agent_id] = agent_memory
    
    def route_query(self, user_input: str) -> str:
        """Route query to appropriate agent."""
        # Use coordinator memory to determine routing
        context = self.coordinator_memory.build_context(user_input)
        
        # Simple routing logic (in practice, use LLM for routing)
        if "python" in user_input.lower():
            return "python_specialist"
        elif "javascript" in user_input.lower():
            return "javascript_specialist"
        else:
            return "general_assistant"
    
    def process_query(self, user_input: str) -> Dict[str, str]:
        """Process query through coordinated agents."""
        # Route to primary agent
        primary_agent_id = self.route_query(user_input)
        primary_agent = self.agents[primary_agent_id]
        
        # Get primary response
        primary_context = primary_agent.build_context(user_input)
        primary_response = f"Primary response from {primary_agent_id}: {user_input}"
        
        # Update coordinator memory
        self.coordinator_memory.observe(user_input, primary_response)
        
        # Get supporting responses from other agents
        supporting_responses = {}
        for agent_id, agent_memory in self.agents.items():
            if agent_id != primary_agent_id:
                agent_context = agent_memory.build_context(user_input)
                supporting_responses[agent_id] = f"Support from {agent_id}: {user_input}"
        
        return {
            "primary_agent": primary_agent_id,
            "primary_response": primary_response,
            "supporting_responses": supporting_responses
        }

# Usage
system = CoordinatedAgentSystem()

# Add agents
system.add_agent("python_specialist", ContextManager(session_id="python"))
system.add_agent("javascript_specialist", ContextManager(session_id="javascript"))
system.add_agent("general_assistant", ContextManager(session_id="general"))

# Process query
result = system.process_query("How do I use Python decorators?")
print(f"Primary: {result['primary_agent']}")
print(f"Response: {result['primary_response']}")
```

## 🔄 Custom Summarization

### Domain-Specific Summarization

```python
from context_manager.memory.summarizer import HierarchicalSummarizer

class TechnicalSummarizer(HierarchicalSummarizer):
    def __init__(self, llm_adapter=None, compression_ratio=0.3):
        super().__init__(llm_adapter, compression_ratio)
    
    def summarize_chunk(self, turns):
        """Custom summarization for technical conversations."""
        combined_text = "\n\n".join([turn.text for turn in turns])
        
        prompt = f"""
Summarize the following technical conversation, focusing on:
1. Technical concepts and solutions discussed
2. Code examples and implementations
3. Best practices mentioned
4. Tools and technologies referenced
5. Problem-solving approaches used

Conversation:
{combined_text}

Technical Summary:"""
        
        try:
            summary = self.llm_adapter.generate_sync(prompt)
            return summary.strip()
        except Exception as e:
            # Fallback to parent implementation
            return super().summarize_chunk(turns)

class PlanningSummarizer(HierarchicalSummarizer):
    def __init__(self, llm_adapter=None, compression_ratio=0.3):
        super().__init__(llm_adapter, compression_ratio)
    
    def summarize_chunk(self, turns):
        """Custom summarization for planning conversations."""
        combined_text = "\n\n".join([turn.text for turn in turns])
        
        prompt = f"""
Summarize the following planning conversation, focusing on:
1. Goals and objectives discussed
2. Timeline and deadlines mentioned
3. Resources and requirements identified
4. Action items and next steps
5. Risks and considerations raised

Conversation:
{combined_text}

Planning Summary:"""
        
        try:
            summary = self.llm_adapter.generate_sync(prompt)
            return summary.strip()
        except Exception as e:
            return super().summarize_chunk(turns)
```

## 📊 Advanced Monitoring

### Custom Memory Analytics

```python
from artiik import ContextManager
from collections import Counter
import re
from typing import Dict, List

class MemoryAnalytics:
    def __init__(self, cm: ContextManager):
        self.cm = cm
    
    def get_memory_insights(self) -> Dict:
        """Get comprehensive memory insights."""
        stats = self.cm.get_stats()
        
        # Get all memories
        all_memories = self.cm.query_memory("", k=1000)
        all_text = " ".join([text for text, _ in all_memories])
        
        return {
            "basic_stats": stats,
            "topics": self._extract_topics(all_text),
            "sentiment": self._analyze_sentiment(all_text),
            "entities": self._extract_entities(all_text),
            "memory_distribution": self._analyze_memory_distribution(all_memories)
        }
    
    def _extract_topics(self, text: str) -> Dict[str, int]:
        """Extract common topics from memory."""
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Filter common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        topics = {word: count for word, count in word_freq.items() 
                 if word not in common_words and len(word) > 3}
        
        return dict(topics.most_common(10))
    
    def _analyze_sentiment(self, text: str) -> Dict:
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
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities."""
        entities = {
            "technologies": re.findall(r'\b(Python|JavaScript|React|Docker|AWS|Azure)\b', text, re.IGNORECASE),
            "numbers": re.findall(r'\$\d+', text),
            "dates": re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text),
        }
        
        return {k: list(set(v)) for k, v in entities.items()}
    
    def _analyze_memory_distribution(self, memories: List) -> Dict:
        """Analyze distribution of memory scores."""
        scores = [score for _, score in memories]
        
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
    cm.ingest_text(memory)

# Get insights
insights = analytics.get_memory_insights()
print("Memory Insights:", insights)
```

## 🚀 Performance Optimization

### Caching Layer

```python
from artiik import ContextManager
from functools import lru_cache
import hashlib
from typing import Optional

class CachedContextManager(ContextManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
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
            self.cache_hits += 1
            return self.context_cache[cache_key]
        
        self.cache_misses += 1
        context = super().build_context(user_input)
        self.context_cache[cache_key] = context
        
        # Limit cache size
        if len(self.context_cache) > 100:
            # Remove oldest entries
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]
        
        return context
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.context_cache)
        }

# Usage
cached_cm = CachedContextManager()

# Build contexts (first will miss, subsequent similar queries will hit)
context1 = cached_cm.build_context("What is Python?")
context2 = cached_cm.build_context("What is Python?")  # Cache hit

stats = cached_cm.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

### Async Context Building

```python
import asyncio
from artiik import ContextManager
from typing import List

class AsyncContextManager(ContextManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def build_context_async(self, user_input: str) -> str:
        """Async version of build_context."""
        # Run context building in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.build_context, user_input)
    
    async def build_contexts_batch(self, user_inputs: List[str]) -> List[str]:
        """Build multiple contexts concurrently."""
        tasks = [self.build_context_async(input_text) for input_text in user_inputs]
        return await asyncio.gather(*tasks)
    
    async def query_memory_async(self, query: str, k: int = 5):
        """Async version of query_memory."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query_memory, query, k)

# Usage
async def main():
    async_cm = AsyncContextManager()
    
    # Build single context
    context = await async_cm.build_context_async("What is Python?")
    
    # Build multiple contexts concurrently
    inputs = ["What is Python?", "What is JavaScript?", "What is React?"]
    contexts = await async_cm.build_contexts_batch(inputs)
    
    # Query memory asynchronously
    results = await async_cm.query_memory_async("programming languages")

# Run async example
# asyncio.run(main())
```

## 🔧 Production Deployment

### Health Monitoring

```python
from artiik import ContextManager
import time
import psutil
import os

class HealthMonitor:
    def __init__(self, cm: ContextManager):
        self.cm = cm
        self.start_time = time.time()
    
    def get_health_status(self) -> Dict:
        """Get comprehensive health status."""
        stats = self.cm.get_stats()
        process = psutil.Process(os.getpid())
        
        return {
            "uptime": time.time() - self.start_time,
            "memory_usage": {
                "process_memory_mb": process.memory_info().rss / 1024**2,
                "stm_utilization": stats['short_term_memory']['utilization'],
                "ltm_entries": stats['long_term_memory']['num_entries']
            },
            "performance": {
                "context_build_time": self._measure_context_build_time(),
                "memory_query_time": self._measure_query_time()
            },
            "status": "healthy" if self._is_healthy() else "degraded"
        }
    
    def _measure_context_build_time(self) -> float:
        """Measure context building performance."""
        start_time = time.time()
        self.cm.build_context("test query")
        return time.time() - start_time
    
    def _measure_query_time(self) -> float:
        """Measure memory query performance."""
        start_time = time.time()
        self.cm.query_memory("test", k=5)
        return time.time() - start_time
    
    def _is_healthy(self) -> bool:
        """Determine if system is healthy."""
        stats = self.cm.get_stats()
        
        # Check memory utilization
        if stats['short_term_memory']['utilization'] > 0.95:
            return False
        
        # Check performance
        if self._measure_context_build_time() > 1.0:  # More than 1 second
            return False
        
        return True

# Usage
cm = ContextManager()
monitor = HealthMonitor(cm)

# Get health status
health = monitor.get_health_status()
print(f"Status: {health['status']}")
print(f"Memory usage: {health['memory_usage']['process_memory_mb']:.1f} MB")
print(f"Context build time: {health['performance']['context_build_time']:.3f}s")
```

---

**Need API details?** → [API Reference](./api_reference.md) | **Troubleshooting?** → [Troubleshooting](./troubleshooting.md) 