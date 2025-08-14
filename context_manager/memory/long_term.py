"""
Long-term memory implementation using FAISS for vector storage.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss
from dataclasses import dataclass
from loguru import logger
from ..llm.embeddings import EmbeddingProvider
from ..core.config import Config
import os
import json


@dataclass
class MemoryEntry:
    """Represents a memory entry in long-term memory."""
    id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    
    def __post_init__(self):
        """Ensure embedding is a numpy array."""
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)


class LongTermMemory:
    """Long-term memory using FAISS for vector similarity search."""
    
    def __init__(self, dimension: int = 384, embedding_provider: Optional[EmbeddingProvider] = None, config: Optional[Config] = None):
        """
        Initialize long-term memory.
        
        Args:
            dimension: Embedding dimension
            embedding_provider: Embedding provider instance
        """
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self._config = config or Config()
        # Prefer auto-detected dimension from embedding provider when available
        provider_dim = getattr(self.embedding_provider, "dimension", None) or self.embedding_provider.get_dimension()
        self.dimension = int(provider_dim) if provider_dim else int(dimension)
        
        # Initialize FAISS index
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors for HNSW
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 100
        
        # Store memory entries
        self.entries: List[MemoryEntry] = []
        self.entry_id_counter = 0
        
        logger.info(f"Initialized LTM with FAISS index (dimension: {self.dimension})")
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory entry to long-term memory.
        
        Args:
            text: Text to store
            metadata: Optional metadata
            
        Returns:
            Memory entry ID
        """
        # Generate embedding and L2-normalize for cosine similarity
        embedding = self.embedding_provider.embed_single(text)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        # Validate embedding dimension
        if embedding.shape[-1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[-1]} does not match index dimension {self.dimension}"
            )
        
        # Create memory entry
        import time
        entry_id = f"mem_{self.entry_id_counter}"
        entry = MemoryEntry(
            id=entry_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # Add to storage
        self.entries.append(entry)
        self.entry_id_counter += 1
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        logger.debug(f"Added memory entry {entry_id} to LTM")
        return entry_id
    
    def search(self, query: str, k: int = 7) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for similar memories.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (memory_entry, similarity_score) tuples
        """
        if not self.entries:
            return []
        
        # Generate query embedding and normalize
        query_embedding = self.embedding_provider.embed_single(query)
        qnorm = np.linalg.norm(query_embedding)
        if qnorm > 0:
            query_embedding = query_embedding / qnorm
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        
        # Convert squared L2 distances over normalized vectors to cosine similarity proxy
        # For unit-normalized vectors FAISS returns squared L2 distances (d2): cos = 1 - d2/2
        d2 = distances[0]
        similarities = 1.0 - (d2 / 2.0)
        # Numerical safety
        similarities = np.clip(similarities, -1.0, 1.0)
        
        # Get results and apply weighted ranking (similarity + recency + importance)
        results: List[Tuple[MemoryEntry, float]] = []
        now = __import__('time').time()
        sim_w = float(self._config.memory.similarity_weight)
        rec_w = float(self._config.memory.recency_weight)
        imp_w = float(self._config.memory.importance_weight)
        half_life = float(self._config.memory.recency_half_life_seconds)

        def recency_score(ts: float) -> float:
            # Exponential decay with half-life
            age = max(0.0, now - float(ts))
            # score in [0,1], newer -> closer to 1
            return 0.5 ** (age / half_life) if half_life > 0 else 0.0

        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.entries):
                entry = self.entries[idx]
                # importance from metadata, default 0..1; allow numeric, else 0
                importance = 0.0
                try:
                    importance = float(entry.metadata.get("importance", 0.0))
                except Exception:
                    importance = 0.0
                rscore = recency_score(entry.timestamp)
                final_score = sim_w * float(similarity) + rec_w * rscore + imp_w * importance
                results.append((entry, float(final_score)))
        
        # Sort by final score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"LTM search returned {len(results)} results for query: {query[:50]}...")
        return results
    
    def search_by_embedding(self, query_embedding: np.ndarray, k: int = 7) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for similar memories using a pre-computed embedding.
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            
        Returns:
            List of (memory_entry, similarity_score) tuples
        """
        if not self.entries:
            return []
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        
        # Convert squared L2 distances over normalized vectors to cosine similarity proxy
        d2 = distances[0]
        similarities = 1.0 - (d2 / 2.0)
        similarities = np.clip(similarities, -1.0, 1.0)
        
        # Get results
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.entries):
                entry = self.entries[idx]
                results.append((entry, float(similarity)))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Get a memory entry by ID.
        
        Args:
            memory_id: Memory entry ID
            
        Returns:
            Memory entry or None if not found
        """
        for entry in self.entries:
            if entry.id == memory_id:
                return entry
        return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: Memory entry ID
            
        Returns:
            True if deleted, False if not found
        """
        for i, entry in enumerate(self.entries):
            if entry.id == memory_id:
                # Remove from entries list
                self.entries.pop(i)
                
                # Rebuild FAISS index (FAISS doesn't support deletion)
                self._rebuild_index()
                
                logger.debug(f"Deleted memory entry {memory_id}")
                return True
        
        return False
    
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current entries."""
        if not self.entries:
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            return
        
        # Create new index
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 100
        
        # Add all embeddings
        embeddings = np.array([entry.embedding for entry in self.entries], dtype=np.float32)
        self.index.add(embeddings)
        
        logger.debug(f"Rebuilt FAISS index with {len(self.entries)} entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about LTM."""
        return {
            "num_entries": len(self.entries),
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            "dimension": self.dimension
        }
    
    def clear(self) -> None:
        """Clear all memory entries."""
        self.entries.clear()
        self.entry_id_counter = 0
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        logger.debug("Cleared LTM")
    
    def __len__(self) -> int:
        """Get number of memory entries."""
        return len(self.entries) 

    # Persistence API
    def save(self, directory_path: str) -> None:
        """Persist FAISS index and entries to disk.
        Writes two files into the directory:
          - index.faiss: FAISS index
          - entries.json: JSON list of entries (id, text, metadata, timestamp)
        """
        os.makedirs(directory_path, exist_ok=True)
        index_path = os.path.join(directory_path, "index.faiss")
        entries_path = os.path.join(directory_path, "entries.json")

        try:
            faiss.write_index(self.index, index_path)
        except Exception as e:
            logger.error(f"Failed to write FAISS index: {e}")
            raise

        try:
            serializable_entries: List[Dict[str, Any]] = [
                {
                    "id": entry.id,
                    "text": entry.text,
                    "metadata": entry.metadata,
                    "timestamp": entry.timestamp,
                }
                for entry in self.entries
            ]
            payload = {
                "dimension": self.dimension,
                "entries": serializable_entries,
                "entry_id_counter": self.entry_id_counter,
                "index_size": int(self.index.ntotal) if hasattr(self.index, 'ntotal') else len(self.entries),
            }
            with open(entries_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.error(f"Failed to write entries JSON: {e}")
            raise

        logger.info(f"Saved LTM to {directory_path} (entries={len(self.entries)}, index={self.index.ntotal})")

    def load(self, directory_path: str) -> None:
        """Load FAISS index and entries from disk (created by save())."""
        index_path = os.path.join(directory_path, "index.faiss")
        entries_path = os.path.join(directory_path, "entries.json")

        if not os.path.exists(index_path) or not os.path.exists(entries_path):
            raise FileNotFoundError("Missing index.faiss or entries.json in provided directory")

        try:
            loaded_index = faiss.read_index(index_path)
        except Exception as e:
            logger.error(f"Failed to read FAISS index: {e}")
            raise

        try:
            with open(entries_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded_dimension = int(payload.get("dimension", self.dimension))
            loaded_entries = payload.get("entries", [])
            loaded_counter = int(payload.get("entry_id_counter", len(loaded_entries)))
        except Exception as e:
            logger.error(f"Failed to read entries JSON: {e}")
            raise

        # Validate and set
        self.dimension = loaded_dimension
        self.index = loaded_index
        self.entries = [
            MemoryEntry(
                id=e["id"],
                text=e["text"],
                embedding=np.zeros(self.dimension, dtype=np.float32),  # placeholder; vectors live in FAISS
                metadata=e.get("metadata", {}),
                timestamp=float(e.get("timestamp", 0.0)),
            )
            for e in loaded_entries
        ]
        self.entry_id_counter = loaded_counter

        # Basic consistency check
        if hasattr(self.index, 'ntotal') and int(self.index.ntotal) != len(self.entries):
            logger.warning(
                f"Loaded index size ({int(self.index.ntotal)}) does not match entries count ({len(self.entries)})."
            )

        logger.info(
            f"Loaded LTM from {directory_path} (entries={len(self.entries)}, index={getattr(self.index, 'ntotal', 0)})"
        )