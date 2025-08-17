"""
Context Orchestration Engine - The "brainstem" of ContextManager.
"""

from typing import List, Optional, Dict, Any, Tuple
from loguru import logger
import time

from .config import Config
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import LongTermMemory
from ..memory.summarizer import HierarchicalSummarizer
from ..llm.adapters import LLMAdapter, create_llm_adapter
from ..llm.embeddings import EmbeddingProvider
from ..utils.token_counter import TokenCounter
import re
import pathlib
from typing import Iterable


class ContextManager:
    """Main Context Orchestration Engine."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        allow_cross_session: bool = False,
        allow_cross_task: bool = False,
    ):
        """
        Initialize ContextManager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        # Scoping
        self.session_id = session_id
        self.task_id = task_id
        self.allow_cross_session = allow_cross_session
        self.allow_cross_task = allow_cross_task
        
        # Initialize components
        self.token_counter = TokenCounter(self.config.llm.model)
        self.embedding_provider = EmbeddingProvider()
        self.llm_adapter = create_llm_adapter(
            provider=self.config.llm.provider,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model
        )
        
        # Initialize memory components
        self.short_term_memory = ShortTermMemory(
            max_tokens=self.config.memory.stm_capacity,
            token_counter=self.token_counter
        )
        
        # Auto-align LTM dimension with embedding provider when available
        provider_dim = self.embedding_provider.get_dimension()
        if provider_dim and provider_dim != self.config.vector_store.dimension:
            logger.warning(
                f"Vector store dimension ({self.config.vector_store.dimension}) does not match embedding model dimension ({provider_dim}). "
                "Using embedding model dimension. Consider updating Config.vector_store.dimension."
            )
        ltm_dimension = provider_dim or self.config.vector_store.dimension
        self.long_term_memory = LongTermMemory(
            dimension=ltm_dimension,
            embedding_provider=self.embedding_provider,
            config=self.config,
        )
        
        self.summarizer = HierarchicalSummarizer(
            llm_adapter=self.llm_adapter,
            compression_ratio=self.config.memory.summary_compression_ratio
        )
        
        logger.info("ContextManager initialized successfully")
    
    def observe(self, user_input: str, assistant_response: str) -> None:
        """
        Observe a new conversation turn.
        
        Args:
            user_input: User's input
            assistant_response: Assistant's response
        """
        # Add to short-term memory
        self.short_term_memory.add_turn(user_input, assistant_response)
        
        # Check if we need to summarize and offload
        if self.short_term_memory.current_tokens > self.config.memory.stm_capacity:
            self._summarize_and_offload()
        
        logger.debug(f"Observed turn. STM tokens: {self.short_term_memory.current_tokens}")
    
    def _summarize_and_offload(self) -> None:
        """Summarize oldest chunk and move to long-term memory."""
        # Get chunk for summarization
        chunk, chunk_tokens = self.short_term_memory.get_chunk_for_summarization(
            self.config.memory.chunk_size
        )
        
        if not chunk:
            return
        
        # Generate summary
        summary = self.summarizer.summarize_chunk(chunk)
        
        # Create metadata
        metadata = self.summarizer.create_summary_metadata(chunk, summary)
        if self.session_id is not None:
            metadata["session_id"] = self.session_id
        if self.task_id is not None:
            metadata["task_id"] = self.task_id
        
        # Add to long-term memory
        memory_id = self.long_term_memory.add_memory(summary, metadata)
        
        logger.info(f"Summarized and offloaded chunk to LTM. Memory ID: {memory_id}")
    
    def build_context(self, user_input: str) -> str:
        """
        Build optimized context for LLM call.
        
        Args:
            user_input: Current user input
            
        Returns:
            Optimized context string
        """
        # Get recent turns from STM
        recent_turns = self.short_term_memory.get_recent_turns(self.config.memory.recent_k)
        recent_texts = [turn.text for turn in recent_turns]
        
        # Search LTM for relevant memories
        ltm_results = self.long_term_memory.search(user_input, k=self.config.memory.ltm_hits_k)
        # Apply scope filters and similarity threshold
        scoped_results = self._filter_ltm_results_by_scope(ltm_results)
        ltm_texts = [entry.text for entry, score in scoped_results if score > 0.5]
        
        # Assemble with budget-aware pruning
        full_context = self._assemble_and_optimize_context(recent_texts, ltm_texts, user_input)
        
        logger.debug(f"Built context with {len(recent_texts)} recent turns and {len(ltm_texts)} LTM hits")
        return full_context
    def _filter_ltm_results_by_scope(self, results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Filter LTM results by current session/task scope if isolation is enabled."""
        filtered = []
        for entry, score in results:
            ok = True
            if self.session_id is not None and not self.allow_cross_session:
                ok = ok and (entry.metadata.get("session_id") == self.session_id)
            if self.task_id is not None and not self.allow_cross_task:
                ok = ok and (entry.metadata.get("task_id") == self.task_id)
            if ok:
                filtered.append((entry, score))
        return filtered

    def set_session(self, session_id: Optional[str], allow_cross_session: Optional[bool] = None) -> None:
        self.session_id = session_id
        if allow_cross_session is not None:
            self.allow_cross_session = allow_cross_session

    def set_task(self, task_id: Optional[str], allow_cross_task: Optional[bool] = None) -> None:
        self.task_id = task_id
        if allow_cross_task is not None:
            self.allow_cross_task = allow_cross_task

    def _assemble_and_optimize_context(self, recent_texts: List[str], ltm_texts: List[str], user_input: str) -> str:
        """Assemble context parts and prune to fit token budget.
        Strategy: prefer keeping recent turns, prune LTM hits first, then recent turns, finally hard-truncate.
        """
        def render(rt: List[str], lt: List[str]) -> str:
            parts: List[str] = []
            if rt:
                parts.append("Recent conversation:")
                parts.extend(rt)
            if lt:
                parts.append("\nRelevant previous context:")
                parts.extend(lt)
            parts.append(f"\nCurrent user input: {user_input}")
            return "\n\n".join(parts)

        # Initial render
        # Optional lightweight keyword filter (hybrid retrieval): drop obvious non-matching LTM by simple keyword match
        keywords = set([w.lower() for w in re.findall(r"\w+", user_input) if len(w) > 3])
        if keywords:
            filtered_ltm = []
            for t in ltm_texts:
                text_words = set([w.lower() for w in re.findall(r"\w+", t) if len(w) > 3])
                if text_words & keywords:
                    filtered_ltm.append(t)
            current_ltm = filtered_ltm if filtered_ltm else list(ltm_texts)
        else:
            current_ltm = list(ltm_texts)
        current_recent = list(recent_texts)
        context_str = render(current_recent, current_ltm)
        budget = self.config.memory.prompt_token_budget

        # Fast path
        if self.token_counter.count_tokens(context_str) <= budget:
            return context_str

        # Prune LTM hits from the end until within budget
        while current_ltm:
            current_ltm.pop()  # remove least recent/relevant hit last
            context_str = render(current_recent, current_ltm)
            if self.token_counter.count_tokens(context_str) <= budget:
                return context_str

        # If still over, reduce recent turns count
        while len(current_recent) > 0:
            current_recent = current_recent[1:]  # drop oldest recent turn first
            context_str = render(current_recent, current_ltm)
            if self.token_counter.count_tokens(context_str) <= budget:
                return context_str

        # Fallback: hard truncate
        return self.token_counter.truncate_to_tokens(context_str, budget)
    
    def query_memory(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Query long-term memory for relevant information.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (text, similarity_score) tuples
        """
        results = self.long_term_memory.search(query, k=k)
        return [(entry.text, score) for entry, score in results]

    # ------------------- Ingestion API -------------------
    def ingest_text(self, text: str, source_id: str, **metadata: Any) -> int:
        """
        Ingest raw text into LTM by chunking and adding as memories.

        Args:
            text: Text content to ingest
            source_id: Identifier for this source (e.g., filename, URL)
            **metadata: Additional metadata; session/task scope auto-injected

        Returns:
            Number of chunks ingested
        """
        if not text:
            return 0
        chunk_size = self.config.memory.ingestion_chunk_size
        overlap = self.config.memory.ingestion_chunk_overlap
        chunks = self.token_counter.split_text_into_token_chunks(text, chunk_size, overlap)
        base_meta = {
            "source_type": metadata.pop("source_type", "text"),
            "source_id": source_id,
        }
        # Inject scope
        if self.session_id is not None:
            base_meta["session_id"] = self.session_id
        if self.task_id is not None:
            base_meta["task_id"] = self.task_id
        count = 0
        for idx, chunk in enumerate(chunks):
            md = base_meta.copy()
            md.update(metadata)
            md["chunk_index"] = idx
            self.long_term_memory.add_memory(chunk, md)
            count += 1
        return count

    def ingest_file(self, path: str, **metadata: Any) -> int:
        """
        Ingest a single file (text-based) into LTM.

        Args:
            path: File path
            **metadata: Additional metadata, e.g., importance

        Returns:
            Number of chunks ingested
        """
        p = pathlib.Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Failed to read file {path}: {e}")
        return self.ingest_text(content, source_id=str(p), source_type="file", **metadata)

    def ingest_directory(self, path: str, file_types: Optional[Iterable[str]] = None, recursive: bool = True, **metadata: Any) -> int:
        """
        Ingest a directory of files into LTM.

        Args:
            path: Directory path
            file_types: Optional list of extensions to include (e.g., [".py", ".md"])
            recursive: Recurse into subdirectories
            **metadata: Additional metadata

        Returns:
            Total ingested chunks across files
        """
        p = pathlib.Path(path)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        exts = set([e.lower() for e in (file_types or [])])
        globber = p.rglob("*") if recursive else p.glob("*")
        total = 0
        for f in globber:
            if not f.is_file():
                continue
            if exts and f.suffix.lower() not in exts:
                continue
            try:
                total += self.ingest_file(str(f), **metadata)
            except Exception:
                # Skip unreadable files, continue
                continue
        return total
    
    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Manually add a memory entry.
        
        Args:
            text: Text to remember
            metadata: Optional metadata
            
        Returns:
            Memory entry ID
        """
        md = metadata.copy() if metadata else {}
        if self.session_id is not None and "session_id" not in md:
            md["session_id"] = self.session_id
        if self.task_id is not None and "task_id" not in md:
            md["task_id"] = self.task_id
        return self.long_term_memory.add_memory(text, md)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the context manager."""
        stm_stats = self.short_term_memory.get_stats()
        ltm_stats = self.long_term_memory.get_stats()
        
        return {
            "short_term_memory": stm_stats,
            "long_term_memory": ltm_stats,
            "config": {
                "stm_capacity": self.config.memory.stm_capacity,
                "chunk_size": self.config.memory.chunk_size,
                "recent_k": self.config.memory.recent_k,
                "ltm_hits_k": self.config.memory.ltm_hits_k,
                "prompt_token_budget": self.config.memory.prompt_token_budget
            }
        }

    # Persistence passthroughs
    def save_memory(self, directory_path: str) -> None:
        """Persist long-term memory to disk."""
        self.long_term_memory.save(directory_path)
    
    def load_memory(self, directory_path: str) -> None:
        """Load long-term memory from disk."""
        self.long_term_memory.load(directory_path)
    
    def clear_memory(self) -> None:
        """Clear all memory."""
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        logger.info("Cleared all memory")
    
    def debug_context_building(self, user_input: str) -> Dict[str, Any]:
        """
        Debug context building process.
        
        Args:
            user_input: User input to debug
            
        Returns:
            Debug information
        """
        # Get recent turns
        recent_turns = self.short_term_memory.get_recent_turns(self.config.memory.recent_k)
        recent_texts = [turn.text for turn in recent_turns]
        
        # Search LTM
        ltm_results = self.long_term_memory.search(user_input, k=self.config.memory.ltm_hits_k)
        
        # Build context
        context = self.build_context(user_input)
        
        return {
            "user_input": user_input,
            "recent_turns_count": len(recent_turns),
            "recent_texts": recent_texts,
            "ltm_results_count": len(ltm_results),
            "ltm_results": [(entry.text, score) for entry, score in ltm_results],
            "final_context_length": len(context),
            "final_context_tokens": self.token_counter.count_tokens(context),
            "context_budget": self.config.memory.prompt_token_budget
        } 