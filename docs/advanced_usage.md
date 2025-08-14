# Advanced Usage

This section covers advanced patterns, custom implementations, and optimization techniques for ContextManager.

## 📚 Table of Contents

- [Custom Memory Implementations](#custom-memory-implementations)
- [Performance Optimization](#performance-optimization)
- [Integration Patterns](#integration-patterns)
- [Custom LLM Adapters](#custom-llm-adapters)
- [Advanced Configuration](#advanced-configuration)
- [Monitoring and Analytics](#monitoring-and-analytics)
- [Production Deployment](#production-deployment)

## Custom Memory Implementations

### Custom Short-Term Memory

Implement a custom STM with different eviction strategies.

```python
from context_manager.memory.short_term import ShortTermMemory, Turn
from collections import deque
import time

class LRUShortTermMemory(ShortTermMemory):
    """Short-term memory with LRU eviction strategy."""
    
    def __init__(self, max_tokens: int = 8000, token_counter=None):
        super().__init__(max_tokens, token_counter)
        self.access_times = {}  # Track last access time for each turn
    
    def _evict_if_needed(self):
        """Evict least recently used turns when capacity exceeded."""
        while self.current_tokens > self.max_tokens and self.turns:
            # Find least recently used turn
            oldest_turn_id = min(self.access_times.keys(), 
                                key=lambda x: self.access_times[x])
            
            # Remove from access tracking
            del self.access_times[oldest_turn_id]
            
            # Remove from turns (simplified - in practice, you'd need to track turn IDs)
            oldest_turn = self.turns.popleft()
            self.current_tokens -= oldest_turn.token_count
    
    def get_recent_turns(self, k: int):
        """Get recent turns and update access times."""
        recent_turns = super().get_recent_turns(k)
        
        # Update access times
        current_time = time.time()
        for turn in recent_turns:
            turn_id = id(turn)  # Use object ID as turn identifier
            self.access_times[turn_id] = current_time
        
        return recent_turns

class PriorityShortTermMemory(ShortTermMemory):
    """Short-term memory with priority-based eviction."""
    
    def __init__(self, max_tokens: int = 8000, token_counter=None):
        super().__init__(max_tokens, token_counter)
        self.priorities = {}  # Track priority for each turn
    
    def add_turn_with_priority(self, user_input: str, assistant_response: str, priority: float = 1.0):
        """Add turn with priority."""
        turn = super().add_turn(user_input, assistant_response)
        self.priorities[id(turn)] = priority
    
    def _evict_if_needed(self):
        """Evict lowest priority turns when capacity exceeded."""
        while self.current_tokens > self.max_tokens and self.turns:
            # Find turn with lowest priority
            lowest_priority_id = min(self.priorities.keys(), 
                                   key=lambda x: self.priorities[x])
            
            # Remove from priority tracking
            del self.priorities[lowest_priority_id]
            
            # Remove from turns
            oldest_turn = self.turns.popleft()
            self.current_tokens -= oldest_turn.token_count
```

### Custom Long-Term Memory

Implement a custom LTM with different storage backends.

```python
from context_manager.memory.long_term import LongTermMemory, MemoryEntry
import sqlite3
import json

class SQLiteLongTermMemory(LongTermMemory):
    """Long-term memory using SQLite for persistence."""
    
    def __init__(self, db_path: str = "memory.db", dimension: int = 384, embedding_provider=None):
        super().__init__(dimension, embedding_provider)
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
    
    def add_memory(self, text: str, metadata=None):
        """Add memory to SQLite database."""
        import numpy as np
        
        # Generate embedding
        embedding = self.embedding_provider.embed_single(text)
        
        # Create memory entry
        import time
        entry_id = f"mem_{self.entry_id_counter}"
        self.entry_id_counter += 1
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?, ?)",
                (
                    entry_id,
                    text,
                    embedding.tobytes(),
                    json.dumps(metadata or {}),
                    time.time()
                )
            )
        
        # Also add to in-memory index for fast search
        entry = MemoryEntry(
            id=entry_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time()
        )
        self.entries.append(entry)
        self.index.add(embedding.reshape(1, -1))
        
        return entry_id
    
    def search(self, query: str, k: int = 7):
        """Search memories using both FAISS and SQLite."""
        # Use FAISS for fast similarity search
        faiss_results = super().search(query, k)
        
        # Also search by text content in SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT text, metadata FROM memories WHERE text LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", k)
            )
            sqlite_results = cursor.fetchall()
        
        # Combine and deduplicate results
        combined_results = []
        seen_texts = set()
        
        # Add FAISS results
        for entry, score in faiss_results:
            if entry.text not in seen_texts:
                combined_results.append((entry, score))
                seen_texts.add(entry.text)
        
        # Add SQLite results
        for text, metadata in sqlite_results:
            if text not in seen_texts:
                # Create temporary entry
                temp_entry = MemoryEntry(
                    id="temp",
                    text=text,
                    embedding=np.zeros(self.dimension),  # Placeholder
                    metadata=json.loads(metadata),
                    timestamp=0
                )
                combined_results.append((temp_entry, 0.5))  # Default score
                seen_texts.add(text)
        
        return combined_results[:k]
    
    def get_stats(self):
        """Get statistics including database info."""
        stats = super().get_stats()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            db_count = cursor.fetchone()[0]
        
        stats["database_entries"] = db_count
        stats["database_path"] = self.db_path
        
        return stats
```

### Custom Summarizer

Implement a custom summarizer with domain-specific prompts.

```python
from context_manager.memory.summarizer import HierarchicalSummarizer
from typing import List, Dict, Any

class DomainSpecificSummarizer(HierarchicalSummarizer):
    """Summarizer with domain-specific prompts."""
    
    def __init__(self, llm_adapter=None, compression_ratio=0.3, domain="general"):
        super().__init__(llm_adapter, compression_ratio)
        self.domain = domain
        self._setup_domain_prompts()
    
    def _setup_domain_prompts(self):
        """Setup domain-specific prompts."""
        self.domain_prompts = {
            "technical": {
                "chunk": """
Summarize this technical conversation, focusing on:
1. Technical concepts and methodologies discussed
2. Code examples, algorithms, or solutions mentioned
3. Tools, frameworks, or technologies referenced
4. Best practices and patterns identified
5. Technical challenges and their resolutions

Conversation:
{text}

Technical Summary:""",
                "hierarchical": """
Create a technical summary of these conversation summaries, highlighting:
1. Overall technical architecture or approach
2. Key technologies and tools used
3. Technical patterns and best practices
4. Problem-solving methodologies
5. Technical insights and lessons learned

Summaries:
{summaries}

Technical Overview:"""
            },
            "business": {
                "chunk": """
Summarize this business conversation, focusing on:
1. Business objectives and goals discussed
2. Strategic decisions and their rationale
3. Financial considerations and budgets
4. Stakeholders and their requirements
5. Action items and next steps

Conversation:
{text}

Business Summary:""",
                "hierarchical": """
Create a business summary of these conversation summaries, highlighting:
1. Overall business strategy and objectives
2. Key decisions and their impact
3. Financial considerations and resource allocation
4. Stakeholder management and requirements
5. Business outcomes and next steps

Summaries:
{summaries}

Business Overview:"""
            },
            "creative": {
                "chunk": """
Summarize this creative conversation, focusing on:
1. Creative concepts and ideas generated
2. Design directions and aesthetic choices
3. Inspiration sources and references
4. Feedback and iteration cycles
5. Final creative decisions and outcomes

Conversation:
{text}

Creative Summary:""",
                "hierarchical": """
Create a creative summary of these conversation summaries, highlighting:
1. Overall creative vision and direction
2. Key creative concepts and themes
3. Design evolution and iterations
4. Inspiration and reference materials
5. Final creative outcomes and next steps

Summaries:
{summaries}

Creative Overview:"""
            }
        }
    
    def summarize_chunk(self, turns, domain=None):
        """Summarize with domain-specific prompt."""
        domain = domain or self.domain
        
        if domain in self.domain_prompts:
            combined_text = "\n\n".join([turn.text for turn in turns])
            prompt = self.domain_prompts[domain]["chunk"].format(text=combined_text)
            
            try:
                summary = self.llm_adapter.generate_sync(
                    prompt,
                    max_tokens=500,
                    temperature=0.3
                )
                return summary.strip()
            except Exception as e:
                print(f"Domain-specific summarization failed: {e}")
                return super().summarize_chunk(turns)
        
        return super().summarize_chunk(turns)
    
    def summarize_hierarchically(self, summaries, domain=None):
        """Create hierarchical summary with domain-specific prompt."""
        domain = domain or self.domain
        
        if domain in self.domain_prompts:
            combined_summaries = "\n\n".join([f"- {summary}" for summary in summaries])
            prompt = self.domain_prompts[domain]["hierarchical"].format(summaries=combined_summaries)
            
            try:
                hierarchical_summary = self.llm_adapter.generate_sync(
                    prompt,
                    max_tokens=800,
                    temperature=0.3
                )
                return hierarchical_summary.strip()
            except Exception as e:
                print(f"Domain-specific hierarchical summarization failed: {e}")
                return super().summarize_hierarchically(summaries)
        
        return super().summarize_hierarchically(summaries)

# Usage
from context_manager.llm.adapters import create_llm_adapter

llm_adapter = create_llm_adapter("openai", api_key="your-key")

# Technical summarizer
technical_summarizer = DomainSpecificSummarizer(llm_adapter, domain="technical")

# Business summarizer
business_summarizer = DomainSpecificSummarizer(llm_adapter, domain="business")

# Creative summarizer
creative_summarizer = DomainSpecificSummarizer(llm_adapter, domain="creative")
```

## Performance Optimization

### Async Context Building

Implement asynchronous context building for better performance.

```python
import asyncio
from context_manager import ContextManager
from context_manager.llm.adapters import create_llm_adapter

class AsyncContextManager(ContextManager):
    """ContextManager with async support."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.llm_adapter = create_llm_adapter(
            self.config.llm.provider,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model
        )
    
    async def build_context_async(self, user_input: str) -> str:
        """Build context asynchronously."""
        # Get recent turns (fast, synchronous)
        recent_turns = self.short_term_memory.get_recent_turns(self.config.memory.recent_k)
        recent_texts = [turn.text for turn in recent_turns]
        
        # Search LTM asynchronously
        ltm_task = asyncio.create_task(self._search_ltm_async(user_input))
        
        # Wait for LTM search
        ltm_results = await ltm_task
        ltm_texts = [entry.text for entry, score in ltm_results if score > 0.5]
        
        # Assemble context
        context_parts = []
        if recent_texts:
            context_parts.append("Recent conversation:")
            context_parts.extend(recent_texts)
        
        if ltm_texts:
            context_parts.append("\nRelevant previous context:")
            context_parts.extend(ltm_texts)
        
        context_parts.append(f"\nCurrent user input: {user_input}")
        full_context = "\n\n".join(context_parts)
        
        # Truncate if necessary
        if self.token_counter.count_tokens(full_context) > self.config.memory.prompt_token_budget:
            full_context = self.token_counter.truncate_to_tokens(
                full_context, 
                self.config.memory.prompt_token_budget
            )
        
        return full_context
    
    async def _search_ltm_async(self, query: str):
        """Search LTM asynchronously."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.long_term_memory.search, 
            query, 
            self.config.memory.ltm_hits_k
        )
    
    async def observe_async(self, user_input: str, assistant_response: str):
        """Observe interaction asynchronously."""
        # Add to STM (fast, synchronous)
        self.short_term_memory.add_turn(user_input, assistant_response)
        
        # Check if summarization is needed
        if self.short_term_memory.current_tokens > self.config.memory.stm_capacity:
            # Run summarization in background
            asyncio.create_task(self._summarize_and_offload_async())
    
    async def _summarize_and_offload_async(self):
        """Summarize and offload asynchronously."""
        # Get chunk for summarization
        chunk, chunk_tokens = self.short_term_memory.get_chunk_for_summarization(
            self.config.memory.chunk_size
        )
        
        if not chunk:
            return
        
        # Generate summary asynchronously
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None,
            self.summarizer.summarize_chunk,
            chunk
        )
        
        # Create metadata
        metadata = self.summarizer.create_summary_metadata(chunk, summary)
        
        # Add to LTM
        memory_id = self.long_term_memory.add_memory(summary, metadata)

# Usage
async def async_agent_example():
    cm = AsyncContextManager()
    
    # Build context asynchronously
    context = await cm.build_context_async("What did we discuss about Python?")
    
    # Generate response (simplified)
    response = f"Response to: {context[:100]}..."
    
    # Observe asynchronously
    await cm.observe_async("What did we discuss about Python?", response)
    
    return response

# Run async example
# asyncio.run(async_agent_example())
```

### Batch Processing

Implement batch processing for multiple queries.

```python
from context_manager import ContextManager
from typing import List, Tuple
import concurrent.futures

class BatchContextManager(ContextManager):
    """ContextManager with batch processing capabilities."""
    
    def __init__(self, config=None, max_workers=4):
        super().__init__(config)
        self.max_workers = max_workers
    
    def build_context_batch(self, user_inputs: List[str]) -> List[str]:
        """Build context for multiple inputs in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.build_context, input_text) 
                      for input_text in user_inputs]
            contexts = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return contexts
    
    def query_memory_batch(self, queries: List[str], k: int = 5) -> List[List[Tuple[str, float]]]:
        """Query memory for multiple queries in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.query_memory, query, k) 
                      for query in queries]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results
    
    def add_memory_batch(self, memories: List[Tuple[str, dict]]) -> List[str]:
        """Add multiple memories in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.add_memory, text, metadata) 
                      for text, metadata in memories]
            memory_ids = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return memory_ids

# Usage
batch_cm = BatchContextManager(max_workers=4)

# Batch context building
inputs = [
    "What did we discuss about Python?",
    "Tell me about our budget planning",
    "What tools did we mention?",
    "Remind me of our timeline"
]

contexts = batch_cm.build_context_batch(inputs)
for input_text, context in zip(inputs, contexts):
    print(f"Input: {input_text}")
    print(f"Context length: {len(context)}")
    print()

# Batch memory querying
queries = ["Python", "budget", "tools", "timeline"]
results = batch_cm.query_memory_batch(queries, k=3)

for query, result in zip(queries, results):
    print(f"Query: {query}")
    for text, score in result:
        print(f"  Score {score:.2f}: {text[:50]}...")
    print()
```

## Integration Patterns

### Flask Web Application

Integrate ContextManager with a Flask web application.

```python
from flask import Flask, request, jsonify
from context_manager import ContextManager
from context_manager.llm.adapters import create_llm_adapter
import os

app = Flask(__name__)

# Global context manager
cm = ContextManager()
llm_adapter = create_llm_adapter("openai", api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with memory."""
    data = request.get_json()
    user_input = data.get('message', '')
    user_id = data.get('user_id', 'default')
    
    # Build context
    context = cm.build_context(user_input)
    
    # Generate response
    response = llm_adapter.generate_sync(
        f"User: {context}\n\nAssistant:",
        max_tokens=500
    )
    
    # Observe interaction
    cm.observe(user_input, response)
    
    return jsonify({
        'response': response,
        'context_length': len(context),
        'memory_stats': cm.get_stats()
    })

@app.route('/memory/query', methods=['POST'])
def query_memory():
    """Query memory endpoint."""
    data = request.get_json()
    query = data.get('query', '')
    k = data.get('k', 5)
    
    results = cm.query_memory(query, k=k)
    
    return jsonify({
        'results': [{'text': text, 'score': score} for text, score in results]
    })

@app.route('/memory/add', methods=['POST'])
def add_memory():
    """Add memory endpoint."""
    data = request.get_json()
    text = data.get('text', '')
    metadata = data.get('metadata', {})
    
    memory_id = cm.add_memory(text, metadata)
    
    return jsonify({
        'memory_id': memory_id,
        'success': True
    })

@app.route('/memory/stats', methods=['GET'])
def get_stats():
    """Get memory statistics."""
    stats = cm.get_stats()
    return jsonify(stats)

@app.route('/memory/clear', methods=['POST'])
def clear_memory():
    """Clear all memory."""
    cm.clear_memory()
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI Application

Integrate with FastAPI for modern async web applications.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from context_manager import ContextManager
from context_manager.llm.adapters import create_llm_adapter
import asyncio
import os

app = FastAPI(title="ContextManager API")

# Global context manager
cm = ContextManager()
llm_adapter = create_llm_adapter("openai", api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    context_length: int
    memory_stats: dict

class MemoryQueryRequest(BaseModel):
    query: str
    k: int = 5

class MemoryQueryResponse(BaseModel):
    results: list

class MemoryAddRequest(BaseModel):
    text: str
    metadata: dict = {}

class MemoryAddResponse(BaseModel):
    memory_id: str
    success: bool

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with memory."""
    try:
        # Build context
        context = cm.build_context(request.message)
        
        # Generate response
        response = llm_adapter.generate_sync(
            f"User: {context}\n\nAssistant:",
            max_tokens=500
        )
        
        # Observe interaction
        cm.observe(request.message, response)
        
        return ChatResponse(
            response=response,
            context_length=len(context),
            memory_stats=cm.get_stats()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/query", response_model=MemoryQueryResponse)
async def query_memory(request: MemoryQueryRequest):
    """Query memory endpoint."""
    try:
        results = cm.query_memory(request.query, k=request.k)
        
        return MemoryQueryResponse(
            results=[{'text': text, 'score': score} for text, score in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/add", response_model=MemoryAddResponse)
async def add_memory(request: MemoryAddRequest):
    """Add memory endpoint."""
    try:
        memory_id = cm.add_memory(request.text, request.metadata)
        
        return MemoryAddResponse(
            memory_id=memory_id,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats")
async def get_stats():
    """Get memory statistics."""
    try:
        return cm.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/clear")
async def clear_memory():
    """Clear all memory."""
    try:
        cm.clear_memory()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Custom LLM Adapters

### Custom LLM Adapter

Implement a custom LLM adapter for your specific provider.

```python
from context_manager.llm.adapters import LLMAdapter
import requests
import json

class CustomLLMAdapter(LLMAdapter):
    """Custom LLM adapter for your specific provider."""
    
    def __init__(self, api_key: str, model: str = "custom-model", base_url: str = "https://api.custom-llm.com"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        try:
            response = await self._make_request(payload)
            return response["choices"][0]["text"]
        except Exception as e:
            raise Exception(f"Custom LLM generation failed: {e}")
    
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Generate text synchronously."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        try:
            response = self._make_sync_request(payload)
            return response["choices"][0]["text"]
        except Exception as e:
            raise Exception(f"Custom LLM generation failed: {e}")
    
    async def _make_request(self, payload: dict) -> dict:
        """Make async HTTP request."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/completions",
                headers=self.headers,
                json=payload
            ) as response:
                return await response.json()
    
    def _make_sync_request(self, payload: dict) -> dict:
        """Make synchronous HTTP request."""
        response = requests.post(
            f"{self.base_url}/v1/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

# Usage
custom_adapter = CustomLLMAdapter(
    api_key="your-custom-api-key",
    model="your-model",
    base_url="https://api.your-llm-provider.com"
)

# Use with ContextManager
from context_manager import Config, LLMConfig

config = Config(
    llm=LLMConfig(
        provider="custom",
        model="your-model",
        api_key="your-custom-api-key"
    )
)

# You would need to modify the factory function to support custom providers
```

## Advanced Configuration

### Dynamic Configuration

Implement dynamic configuration that adapts based on usage patterns.

```python
from context_manager import Config, ContextManager
import time
from collections import defaultdict

class AdaptiveContextManager(ContextManager):
    """ContextManager with adaptive configuration."""
    
    def __init__(self, base_config=None):
        super().__init__(base_config)
        self.usage_stats = defaultdict(int)
        self.performance_metrics = []
        self.last_adaptation = time.time()
        self.adaptation_interval = 300  # 5 minutes
    
    def build_context(self, user_input: str) -> str:
        """Build context with performance tracking."""
        start_time = time.time()
        
        try:
            context = super().build_context(user_input)
            
            # Track performance
            build_time = time.time() - start_time
            self._track_performance(build_time, len(context))
            
            return context
        except Exception as e:
            self._track_error(e)
            raise
    
    def _track_performance(self, build_time: float, context_length: int):
        """Track performance metrics."""
        self.performance_metrics.append({
            'timestamp': time.time(),
            'build_time': build_time,
            'context_length': context_length,
            'stm_turns': len(self.short_term_memory.turns),
            'ltm_entries': len(self.long_term_memory.entries)
        })
        
        # Keep only recent metrics
        if len(self.performance_metrics) > 100:
            self.performance_metrics = self.performance_metrics[-50:]
        
        # Check if adaptation is needed
        if time.time() - self.last_adaptation > self.adaptation_interval:
            self._adapt_configuration()
    
    def _adapt_configuration(self):
        """Adapt configuration based on performance metrics."""
        if not self.performance_metrics:
            return
        
        recent_metrics = self.performance_metrics[-20:]
        avg_build_time = sum(m['build_time'] for m in recent_metrics) / len(recent_metrics)
        avg_context_length = sum(m['context_length'] for m in recent_metrics) / len(recent_metrics)
        
        # Adapt based on performance
        if avg_build_time > 1.0:  # Slow performance
            # Reduce context size
            self.config.memory.recent_k = max(3, self.config.memory.recent_k - 1)
            self.config.memory.ltm_hits_k = max(5, self.config.memory.ltm_hits_k - 1)
            print(f"Adapted: Reduced context size (build_time: {avg_build_time:.2f}s)")
        
        elif avg_build_time < 0.1 and avg_context_length < self.config.memory.prompt_token_budget * 0.5:
            # Fast performance, can increase context
            self.config.memory.recent_k = min(10, self.config.memory.recent_k + 1)
            self.config.memory.ltm_hits_k = min(15, self.config.memory.ltm_hits_k + 1)
            print(f"Adapted: Increased context size (build_time: {avg_build_time:.2f}s)")
        
        self.last_adaptation = time.time()
    
    def _track_error(self, error: Exception):
        """Track errors for adaptation."""
        self.usage_stats['errors'] += 1
        
        # If too many errors, reduce complexity
        if self.usage_stats['errors'] > 10:
            self.config.memory.prompt_token_budget = max(6000, self.config.memory.prompt_token_budget - 1000)
            self.usage_stats['errors'] = 0
            print("Adapted: Reduced token budget due to errors")
    
    def get_adaptation_stats(self) -> dict:
        """Get adaptation statistics."""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = self.performance_metrics[-20:]
        
        return {
            "avg_build_time": sum(m['build_time'] for m in recent_metrics) / len(recent_metrics),
            "avg_context_length": sum(m['context_length'] for m in recent_metrics) / len(recent_metrics),
            "total_metrics": len(self.performance_metrics),
            "errors": self.usage_stats['errors'],
            "last_adaptation": self.last_adaptation
        }

# Usage
adaptive_cm = AdaptiveContextManager()

# Use normally - it will adapt automatically
for i in range(100):
    context = adaptive_cm.build_context(f"Test input {i}")
    adaptive_cm.observe(f"Test input {i}", f"Test response {i}")

# Check adaptation stats
stats = adaptive_cm.get_adaptation_stats()
print("Adaptation stats:", stats)
```

## Monitoring and Analytics

### Comprehensive Monitoring

Implement comprehensive monitoring for production use.

```python
import logging
import time
from context_manager import ContextManager
from typing import Dict, Any, List
import json

class MonitoredContextManager(ContextManager):
    """ContextManager with comprehensive monitoring."""
    
    def __init__(self, config=None, log_file="context_manager.log"):
        super().__init__(config)
        
        # Setup logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history = []
        self.error_history = []
        self.memory_usage_history = []
    
    def build_context(self, user_input: str) -> str:
        """Build context with monitoring."""
        start_time = time.time()
        
        try:
            # Log input
            self.logger.info(f"Building context for input: {user_input[:100]}...")
            
            # Build context
            context = super().build_context(user_input)
            
            # Track performance
            build_time = time.time() - start_time
            self._track_performance(build_time, len(context), user_input)
            
            # Log success
            self.logger.info(f"Context built successfully in {build_time:.3f}s, length: {len(context)}")
            
            return context
            
        except Exception as e:
            # Track error
            self._track_error(e, user_input)
            self.logger.error(f"Context building failed: {e}")
            raise
    
    def observe(self, user_input: str, assistant_response: str):
        """Observe interaction with monitoring."""
        start_time = time.time()
        
        try:
            super().observe(user_input, assistant_response)
            
            # Track memory usage
            stats = self.get_stats()
            self._track_memory_usage(stats)
            
            # Log observation
            observe_time = time.time() - start_time
            self.logger.info(f"Interaction observed in {observe_time:.3f}s")
            
        except Exception as e:
            self._track_error(e, f"observe: {user_input}")
            self.logger.error(f"Observation failed: {e}")
            raise
    
    def _track_performance(self, build_time: float, context_length: int, user_input: str):
        """Track performance metrics."""
        self.performance_history.append({
            'timestamp': time.time(),
            'build_time': build_time,
            'context_length': context_length,
            'input_length': len(user_input),
            'stm_turns': len(self.short_term_memory.turns),
            'ltm_entries': len(self.long_term_memory.entries)
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    def _track_error(self, error: Exception, context: str):
        """Track errors."""
        self.error_history.append({
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context[:200]
        })
        
        # Keep only recent errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]
    
    def _track_memory_usage(self, stats: Dict[str, Any]):
        """Track memory usage."""
        self.memory_usage_history.append({
            'timestamp': time.time(),
            'stm_turns': stats['short_term_memory']['num_turns'],
            'stm_tokens': stats['short_term_memory']['current_tokens'],
            'stm_utilization': stats['short_term_memory']['utilization'],
            'ltm_entries': stats['long_term_memory']['num_entries'],
            'ltm_index_size': stats['long_term_memory']['index_size']
        })
        
        # Keep only recent history
        if len(self.memory_usage_history) > 1000:
            self.memory_usage_history = self.memory_usage_history[-500:]
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_performance = self.performance_history[-100:]
        recent_errors = self.error_history[-20:]
        recent_memory = self.memory_usage_history[-100:]
        
        # Calculate metrics
        avg_build_time = sum(p['build_time'] for p in recent_performance) / len(recent_performance)
        avg_context_length = sum(p['context_length'] for p in recent_performance) / len(recent_performance)
        error_rate = len(recent_errors) / len(recent_performance) if recent_performance else 0
        
        # Memory trends
        if recent_memory:
            avg_stm_utilization = sum(m['stm_utilization'] for m in recent_memory) / len(recent_memory)
            avg_ltm_entries = sum(m['ltm_entries'] for m in recent_memory) / len(recent_memory)
        else:
            avg_stm_utilization = 0
            avg_ltm_entries = 0
        
        return {
            "performance": {
                "avg_build_time": avg_build_time,
                "avg_context_length": avg_context_length,
                "total_operations": len(self.performance_history),
                "recent_operations": len(recent_performance)
            },
            "errors": {
                "error_rate": error_rate,
                "total_errors": len(self.error_history),
                "recent_errors": len(recent_errors),
                "error_types": list(set(e['error_type'] for e in recent_errors))
            },
            "memory": {
                "avg_stm_utilization": avg_stm_utilization,
                "avg_ltm_entries": avg_ltm_entries,
                "current_stm_turns": len(self.short_term_memory.turns),
                "current_ltm_entries": len(self.long_term_memory.entries)
            },
            "configuration": {
                "stm_capacity": self.config.memory.stm_capacity,
                "chunk_size": self.config.memory.chunk_size,
                "recent_k": self.config.memory.recent_k,
                "ltm_hits_k": self.config.memory.ltm_hits_k,
                "prompt_token_budget": self.config.memory.prompt_token_budget
            }
        }
    
    def export_monitoring_data(self, filename: str):
        """Export monitoring data to JSON file."""
        data = {
            "performance_history": self.performance_history,
            "error_history": self.error_history,
            "memory_usage_history": self.memory_usage_history,
            "current_stats": self.get_stats(),
            "monitoring_report": self.get_monitoring_report()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring data exported to {filename}")

# Usage
monitored_cm = MonitoredContextManager()

# Use normally - monitoring happens automatically
for i in range(50):
    try:
        context = monitored_cm.build_context(f"Test input {i}")
        monitored_cm.observe(f"Test input {i}", f"Test response {i}")
    except Exception as e:
        print(f"Error in iteration {i}: {e}")

# Get monitoring report
report = monitored_cm.get_monitoring_report()
print("Monitoring Report:")
print(json.dumps(report, indent=2))

# Export data
monitored_cm.export_monitoring_data("monitoring_data.json")
```

## Production Deployment

### Docker Configuration

Example Docker setup for production deployment.

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  context-manager-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### Kubernetes Configuration

Example Kubernetes deployment.

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: context-manager
spec:
  replicas: 3
  selector:
    matchLabels:
      app: context-manager
  template:
    metadata:
      labels:
        app: context-manager
    spec:
      containers:
      - name: context-manager
        image: context-manager:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: context-manager-service
spec:
  selector:
    app: context-manager
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

**Next**: [Troubleshooting](./troubleshooting.md) → Common issues and solutions 