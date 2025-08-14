"""
Embedding utilities for ContextManager.
"""

from typing import List, Union, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger


class EmbeddingProvider:
    """Provider for text embeddings using sentence-transformers."""
    
    # Available models with their sizes and dimensions
    MODEL_OPTIONS = {
        "all-MiniLM-L6-v2": {
            "size_mb": 90,
            "dimension": 384,
            "speed": "fast",
            "quality": "good"
        },
        "all-mpnet-base-v2": {
            "size_mb": 420,
            "dimension": 768,
            "speed": "medium",
            "quality": "excellent"
        },
        "all-MiniLM-L12-v2": {
            "size_mb": 120,
            "dimension": 384,
            "speed": "fast",
            "quality": "very good"
        },
        "paraphrase-MiniLM-L3-v2": {
            "size_mb": 61,
            "dimension": 384,
            "speed": "very fast",
            "quality": "good"
        }
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding provider.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        self.dimension: Optional[int] = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            if self.model_name in self.MODEL_OPTIONS:
                model_info = self.MODEL_OPTIONS[self.model_name]
                logger.info(f"Model size: {model_info['size_mb']}MB, Dimension: {model_info['dimension']}")
            
            self.model = SentenceTransformer(self.model_name)
            try:
                self.dimension = int(self.model.get_sentence_embedding_dimension())
            except Exception:
                # Fallback to known value from table if available
                self.dimension = self.MODEL_OPTIONS.get(self.model_name, {}).get("dimension", None)
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                logger.info("Attempting to load fallback model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                try:
                    self.dimension = int(self.model.get_sentence_embedding_dimension())
                except Exception:
                    self.dimension = self.MODEL_OPTIONS.get("all-MiniLM-L6-v2", {}).get("dimension", None)
                logger.info("Successfully loaded fallback embedding model: all-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        if self.model_name in self.MODEL_OPTIONS:
            return self.MODEL_OPTIONS[self.model_name].copy()
        return {
            "size_mb": "unknown",
            "dimension": "unknown", 
            "speed": "unknown",
            "quality": "unknown"
        }
    
    def list_available_models(self) -> Dict[str, Dict[str, any]]:
        """List all available models with their specifications."""
        return self.MODEL_OPTIONS.copy()
    
    def get_dimension(self) -> Optional[int]:
        """Return the embedding dimension of the loaded model, if known."""
        return self.dimension
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Text or list of texts to embed
            
        Returns:
            Embeddings as numpy array
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Single embedding as numpy array
        """
        return self.embed([text])[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def batch_similarity(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between query embedding and batch of embeddings.
        
        Args:
            query_embedding: Query embedding
            embeddings: Batch of embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings))
        
        # Normalize all embeddings
        norms = np.linalg.norm(embeddings, axis=1)
        valid_mask = norms > 0
        
        similarities = np.zeros(len(embeddings))
        if np.any(valid_mask):
            similarities[valid_mask] = np.dot(embeddings[valid_mask], query_embedding) / (norms[valid_mask] * query_norm)
        
        return similarities 