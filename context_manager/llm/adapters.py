"""
LLM adapters for different providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import openai
import anthropic
from loguru import logger


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt (synchronous)."""
        pass


class OpenAIAdapter(LLMAdapter):
    """OpenAI LLM adapter."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key
            model: Model name
        """
        self.model = model
        # Synchronous client
        self.client = openai.OpenAI(api_key=api_key)
        # Async client when available (openai >= 1.0.0 provides Async client)
        try:
            self.async_client = openai.AsyncOpenAI(api_key=api_key)
        except Exception:
            self.async_client = None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously."""
        try:
            if self.async_client is None:
                # Fallback: run sync in thread to avoid blocking
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.generate_sync(prompt, **kwargs)
                )
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Generate text synchronously."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise


class AnthropicAdapter(LLMAdapter):
    """Anthropic LLM adapter."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic adapter.
        
        Args:
            api_key: Anthropic API key
            model: Model name
        """
        self.model = model
        # Synchronous client
        self.client = anthropic.Anthropic(api_key=api_key)
        # Async client (Anthropic supports async via anthropic.AsyncAnthropic)
        try:
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        except Exception:
            self.async_client = None
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously."""
        try:
            if self.async_client is None:
                # Fallback: run sync in thread to avoid blocking
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.generate_sync(prompt, **kwargs)
                )
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Generate text synchronously."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise


def create_llm_adapter(provider: str, **kwargs) -> LLMAdapter:
    """
    Factory function to create LLM adapter.
    
    Args:
        provider: LLM provider ("openai" or "anthropic")
        **kwargs: Additional arguments for the adapter
        
    Returns:
        LLM adapter instance
    """
    if provider == "openai":
        return OpenAIAdapter(**kwargs)
    elif provider == "anthropic":
        return AnthropicAdapter(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 