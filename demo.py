#!/usr/bin/env python3
"""
Demo script for ContextManager.
"""

import os
import sys
from loguru import logger

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from context_manager.core import ContextManager, Config
from context_manager.examples.agent_example import SimpleAgent


def demo_context_manager():
    """Demo the ContextManager directly."""
    print("🧠 ContextManager Demo")
    print("=" * 50)
    
    # Initialize ContextManager
    config = Config()
    config.llm.api_key = os.getenv("OPENAI_API_KEY")
    config.debug = True
    
    cm = ContextManager(config)
    
    # Simulate a conversation
    conversation = [
        ("Hello! I'm planning a trip to Japan.", "That sounds exciting! Japan is a beautiful country with rich culture and history."),
        ("I want to visit Tokyo, Kyoto, and Osaka. What should I know?", "Great choices! Tokyo is the modern capital, Kyoto has traditional temples, and Osaka is known for food and nightlife."),
        ("What's the best time to visit?", "Spring (March-May) for cherry blossoms or Fall (October-November) for autumn colors are ideal."),
        ("How much should I budget for 10 days?", "For a comfortable trip, budget around $200-300 per day including accommodation, food, and activities."),
        ("Can you remind me what we discussed about my Japan trip?", "We discussed your plan to visit Tokyo, Kyoto, and Osaka. You asked about the best time to visit (spring or fall) and budgeting around $200-300 per day for 10 days."),
        ("What was my budget calculation again?", "We discussed budgeting $200-300 per day for 10 days, which would be approximately $2,000-3,000 total for your Japan trip.")
    ]
    
    for i, (user_input, assistant_response) in enumerate(conversation, 1):
        print(f"\n👤 Turn {i}: {user_input}")
        print(f"🤖 Assistant: {assistant_response}")
        
        # Observe the interaction
        cm.observe(user_input, assistant_response)
        
        # Show memory stats every few turns
        if i % 3 == 0:
            stats = cm.get_stats()
            print(f"\n📊 Memory Stats:")
            print(f"  STM turns: {stats['short_term_memory']['num_turns']}")
            print(f"  LTM entries: {stats['long_term_memory']['num_entries']}")
    
    # Test memory query
    print(f"\n🔍 Memory Query: 'Japan budget'")
    results = cm.query_memory("Japan budget")
    for text, score in results:
        print(f"  Score {score:.2f}: {text[:100]}...")
    
    # Test context building
    print(f"\n🔧 Context Building for: 'What was my budget again?'")
    debug_info = cm.debug_context_building("What was my budget again?")
    print(f"  Recent turns: {debug_info['recent_turns_count']}")
    print(f"  LTM hits: {debug_info['ltm_results_count']}")
    print(f"  Final context tokens: {debug_info['final_context_tokens']}/{debug_info['context_budget']}")


def demo_agent():
    """Demo the SimpleAgent with ContextManager."""
    print("\n🤖 SimpleAgent Demo with ContextManager")
    print("=" * 50)
    
    # Initialize agent
    config = Config()
    config.llm.api_key = os.getenv("OPENAI_API_KEY")
    
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


def main():
    """Main demo function."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
        print("   Set it with: export OPENAI_API_KEY='your-api-key'")
        print()
    
    try:
        # Demo ContextManager directly
        demo_context_manager()
        
        # Demo with agent
        demo_agent()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        print("This might be due to missing API key or network issues.")


if __name__ == "__main__":
    main() 