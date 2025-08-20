# Examples

This page provides practical examples of how to use ContextManager in real-world scenarios.

## 🚀 Basic Examples

### Simple Chat Agent

```python
from artiik import ContextManager
import openai

# Initialize
cm = ContextManager()
openai.api_key = "your-api-key"

def chat_agent(user_input: str) -> str:
    # Build context from memory
    context = cm.build_context(user_input)
    
    # Call OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": context}],
        max_tokens=500
    )
    
    # Get response
    assistant_response = response.choices[0].message.content
    
    # Store the interaction
    cm.observe(user_input, assistant_response)
    
    return assistant_response

# Usage
response = chat_agent("Hello! I'm planning a trip to Japan.")
print(response)
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

## 🔧 Advanced Examples

### Custom Configuration

```python
from artiik import Config, ContextManager

# Create custom configuration
config = Config(
    memory=MemoryConfig(
        stm_capacity=12000,         # Larger short-term memory
        chunk_size=3000,            # Larger chunks
        recent_k=8,                 # More recent turns
        ltm_hits_k=10,              # More LTM results
        prompt_token_budget=16000,  # Larger context budget
        summary_compression_ratio=0.2,  # More aggressive compression
    ),
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="your-api-key",
        max_tokens=1000,
        temperature=0.3,  # More focused responses
    ),
    debug=True,  # Enable debug logging
)

cm = ContextManager(config)
```

### Session and Task Scoping

```python
from artiik import ContextManager

# Create scoped context managers
user1_cm = ContextManager(session_id="user1")
user2_cm = ContextManager(session_id="user2")

# Each user has isolated memory
user1_cm.observe("I like Python", "Python is great!")
user2_cm.observe("I like JavaScript", "JavaScript is awesome!")

# Queries are scoped to each user
user1_results = user1_cm.query_memory("programming language")
user2_results = user2_cm.query_memory("programming language")

# Cross-session access (if needed)
shared_cm = ContextManager(
    session_id="shared",
    allow_cross_session=True
)
```

### Memory Persistence

```python
from artiik import ContextManager

# Save memory to disk
cm = ContextManager()
cm.observe("Hello", "Hi there!")
cm.observe("How are you?", "I'm doing well!")

# Save memory
cm.save_memory("./memory_backup")

# Load memory in a new instance
new_cm = ContextManager()
new_cm.load_memory("./memory_backup")

# Memory is restored
results = new_cm.query_memory("greeting")
print(results)
```

### Debug Context Building

```python
from artiik import ContextManager

cm = ContextManager()

# Add some conversation
for i in range(10):
    cm.observe(f"User message {i}", f"Assistant response {i}")

# Debug context building
debug_info = cm.debug_context_building("What did we discuss?")
print(f"Recent turns: {debug_info['recent_turns_count']}")
print(f"LTM hits: {debug_info['ltm_results_count']}")
print(f"Final tokens: {debug_info['final_context_tokens']}")
print(f"Context budget: {debug_info['context_budget']}")
```

## 🎯 Use Case Examples

### Long Conversation Management

```python
from artiik import ContextManager

cm = ContextManager()

# Simulate a long conversation
conversation_topics = [
    ("Let's talk about Python", "Python is a versatile language..."),
    ("What about web development?", "Web development involves..."),
    ("Tell me about databases", "Databases store and organize data..."),
    ("How about machine learning?", "Machine learning uses algorithms..."),
    ("What's the weather like?", "I can't check the weather..."),
    ("Back to Python - what are decorators?", "Decorators are functions that modify..."),
]

for user_input, response in conversation_topics:
    cm.observe(user_input, response)

# Even after many turns, context is maintained
context = cm.build_context("Tell me more about what we discussed earlier")
print(context)
```

### Multi-Topic Discussion

```python
from artiik import ContextManager

cm = ContextManager()

# Switch between topics
topics = [
    ("Let's plan a trip to Japan", "Japan is beautiful..."),
    ("What programming languages should I learn?", "Python, JavaScript, and Go are good choices..."),
    ("Tell me about cooking", "Cooking involves various techniques..."),
    ("Back to Japan - what cities should I visit?", "Tokyo, Kyoto, and Osaka are must-visit cities..."),
    ("What about the programming question?", "For beginners, I'd recommend starting with Python..."),
]

for user_input, response in topics:
    cm.observe(user_input, response)

# Context switching works seamlessly
japan_context = cm.build_context("What was my Japan plan?")
programming_context = cm.build_context("What programming advice did you give?")
```

### Information Retrieval

```python
from artiik import ContextManager

cm = ContextManager()

# Add detailed conversation
detailed_conversation = [
    ("I'm planning a 10-day trip to Japan", "That's exciting! Japan has so much to offer."),
    ("I want to visit Tokyo, Kyoto, and Osaka", "Great choices! Each city has unique attractions."),
    ("What's the best time to visit?", "Spring (March-May) for cherry blossoms or Fall (October-November) for autumn colors."),
    ("How much should I budget?", "For a comfortable trip, budget around $200-300 per day including accommodation, food, and activities."),
    ("What about transportation?", "Japan has excellent public transportation. Get a JR Pass for inter-city travel."),
    ("Where should I stay in Tokyo?", "Shinjuku, Shibuya, or Ginza are popular areas with good access to attractions."),
]

for user_input, response in detailed_conversation:
    cm.observe(user_input, response)

# Query specific information
budget_results = cm.query_memory("budget cost", k=3)
transport_results = cm.query_memory("transportation", k=3)
accommodation_results = cm.query_memory("where to stay", k=3)

print("Budget info:", budget_results)
print("Transport info:", transport_results)
print("Accommodation info:", accommodation_results)
```

### Tool-Using Agent

```python
from artiik import ContextManager
import openai

class ToolAgent:
    def __init__(self):
        self.cm = ContextManager()
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
        
        # Generate response (simplified - you'd use your LLM here)
        response = f"Based on the context: {context[:100]}..."
        
        # Store interaction
        self.cm.observe(user_input, response)
        
        return response

# Usage
agent = ToolAgent()
response = agent.respond("What's 15 * 23?")
print(response)
```

## 📊 Monitoring Examples

### Memory Statistics

```python
from artiik import ContextManager

cm = ContextManager()

# Add some conversation
for i in range(5):
    cm.observe(f"User {i}", f"Assistant {i}")

# Get memory statistics
stats = cm.get_stats()

print("Memory Statistics:")
print(f"STM turns: {stats['short_term_memory']['num_turns']}")
print(f"STM tokens: {stats['short_term_memory']['current_tokens']}")
print(f"STM utilization: {stats['short_term_memory']['utilization']:.2%}")
print(f"LTM entries: {stats['long_term_memory']['num_entries']}")
print(f"LTM index size: {stats['long_term_memory']['index_size']}")
```

### Performance Monitoring

```python
import time
from artiik import ContextManager

cm = ContextManager()

# Monitor context building performance
def monitor_performance():
    start_time = time.time()
    context = cm.build_context("Test query")
    build_time = time.time() - start_time
    
    debug_info = cm.debug_context_building("Test query")
    
    print(f"Context build time: {build_time:.3f} seconds")
    print(f"Recent turns: {debug_info['recent_turns_count']}")
    print(f"LTM hits: {debug_info['ltm_results_count']}")
    print(f"Final tokens: {debug_info['final_context_tokens']}")
    print(f"Token utilization: {debug_info['final_context_tokens'] / debug_info['context_budget']:.2%}")

monitor_performance()
```

## 🔧 Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from artiik import ContextManager

app = Flask(__name__)
cm = ContextManager()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    
    # Build context
    context = cm.build_context(user_input)
    
    # Call your LLM here
    response = f"Response to: {user_input}"  # Replace with actual LLM call
    
    # Store interaction
    cm.observe(user_input, response)
    
    return jsonify({'response': response})

@app.route('/memory', methods=['GET'])
def query_memory():
    query = request.args.get('q', '')
    k = int(request.args.get('k', 5))
    
    results = cm.query_memory(query, k=k)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from artiik import ContextManager

app = FastAPI()
cm = ContextManager()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Build context
        context = cm.build_context(request.message)
        
        # Call your LLM here
        response = f"Response to: {request.message}"  # Replace with actual LLM call
        
        # Store interaction
        cm.observe(request.message, response)
        
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory")
async def query_memory(q: str, k: int = 5):
    try:
        results = cm.query_memory(q, k=k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

**Ready for more?** → [Advanced Usage](./advanced_usage.md) | [API Reference](./api_reference.md) 