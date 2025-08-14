"""
Example agent implementation using ContextManager.
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger

from ..core import ContextManager, Config
from ..llm.adapters import create_llm_adapter


class SimpleAgent:
    """
    Example agent that uses ContextManager for memory and context management.
    
    This agent has access to tools but no built-in memory - ContextManager
    handles all the memory and context management automatically.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the agent.
        
        Args:
            config: Configuration for ContextManager
        """
        # Initialize ContextManager
        self.context_manager = ContextManager(config)
        
        # Initialize LLM adapter
        self.llm_adapter = create_llm_adapter(
            provider=self.context_manager.config.llm.provider,
            api_key=self.context_manager.config.llm.api_key,
            model=self.context_manager.config.llm.model
        )
        
        # Define available tools
        self.tools = {
            "search_web": self._search_web,
            "get_weather": self._get_weather,
            "calculate": self._calculate,
            "get_time": self._get_time
        }
        
        logger.info("SimpleAgent initialized with ContextManager")
    
    def _search_web(self, query: str) -> str:
        """Mock web search tool."""
        return f"Search results for '{query}': Found 5 relevant pages about {query}."
    
    def _get_weather(self, location: str) -> str:
        """Mock weather tool."""
        return f"Weather in {location}: 72°F, partly cloudy with 20% chance of rain."
    
    def _calculate(self, expression: str) -> str:
        """Mock calculator tool."""
        try:
            result = eval(expression)
            return f"Calculation result: {expression} = {result}"
        except:
            return f"Error calculating: {expression}"
    
    def _get_time(self) -> str:
        """Mock time tool."""
        import datetime
        return f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    def _build_tool_prompt(self, user_input: str) -> str:
        """Build prompt that includes available tools."""
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        ])
        
        return f"""You are a helpful AI assistant with access to the following tools:

{tool_descriptions}

When the user asks for information that requires these tools, use them appropriately.
Always respond naturally and conversationally.

User: {user_input}"""
    
    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's input
            
        Returns:
            Agent's response
        """
        # Build context using ContextManager
        context = self.context_manager.build_context(user_input)
        
        # Add tool information to context
        full_prompt = self._build_tool_prompt(context)
        
        try:
            # Generate response using LLM
            response = self.llm_adapter.generate_sync(
                full_prompt,
                max_tokens=self.context_manager.config.llm.max_tokens,
                temperature=self.context_manager.config.llm.temperature
            )
            
            # Observe the interaction (this updates memory)
            self.context_manager.observe(user_input, response)
            
            logger.debug(f"Generated response for user input: {user_input[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            self.context_manager.observe(user_input, error_response)
            return error_response
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.context_manager.get_stats()
    
    def query_memory(self, query: str) -> List[tuple]:
        """Query memory for relevant information."""
        return self.context_manager.query_memory(query)
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Manually add a memory entry."""
        return self.context_manager.add_memory(text, metadata)
    
    def debug_context(self, user_input: str) -> Dict[str, Any]:
        """Debug context building process."""
        return self.context_manager.debug_context_building(user_input)


def demo_agent():
    """Demo function showing how to use the agent."""
    # Set up API key (you would normally load this from environment)
    config = Config()
    config.llm.api_key = os.getenv("OPENAI_API_KEY")
    
    # Create agent
    agent = SimpleAgent(config)
    
    # Example conversation
    conversation = [
        "Hello! I'm planning a trip to Japan. Can you help me?",
        "I want to visit Tokyo, Kyoto, and Osaka. What's the weather like there?",
        "I need to calculate my budget. If I spend $150 per day for 10 days, how much is that?",
        "What time is it right now?",
        "Can you remind me what we discussed about my Japan trip?",
        "What was my budget calculation again?"
    ]
    
    print("🤖 SimpleAgent Demo with ContextManager")
    print("=" * 50)
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\n👤 User {i}: {user_input}")
        
        # Get response
        response = agent.respond(user_input)
        print(f"🤖 Agent: {response}")
        
        # Show memory stats every few turns
        if i % 3 == 0:
            stats = agent.get_memory_stats()
            print(f"\n📊 Memory Stats:")
            print(f"  STM turns: {stats['short_term_memory']['num_turns']}")
            print(f"  LTM entries: {stats['long_term_memory']['num_entries']}")
    
    # Show final memory query
    print(f"\n🔍 Memory Query: 'Japan trip budget'")
    results = agent.query_memory("Japan trip budget")
    for text, score in results:
        print(f"  Score {score:.2f}: {text[:100]}...")


if __name__ == "__main__":
    demo_agent() 