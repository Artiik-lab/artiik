from unittest.mock import Mock, patch
import time
import numpy as np

from context_manager.core import ContextManager, Config


def _mock_embedder(dim=8):
    class M:
        def get_sentence_embedding_dimension(self):
            return dim

        def encode(self, texts, convert_to_numpy=True):
            import numpy as _np
            n = len(texts) if isinstance(texts, list) else 1
            return _np.ones((n, dim), dtype=_np.float32)

    return M()


def test_importance_boosts_ranking():
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls, \
         patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
        mock_model_cls.return_value = _mock_embedder(8)
        mock_llm.return_value = Mock()

        cfg = Config()
        # Emphasize importance over similarity/recency
        cfg.memory.similarity_weight = 0.0
        cfg.memory.recency_weight = 0.0
        cfg.memory.importance_weight = 1.0

        cm = ContextManager(cfg)
        cm.add_memory("low importance", {"importance": 0.1})
        cm.add_memory("high importance", {"importance": 0.9})

        results = cm.query_memory("q", k=2)
        # Ensure 'high importance' ranks higher
        assert results and 'high importance' in results[0][0]


def test_recency_boosts_ranking():
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls, \
         patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
        mock_model_cls.return_value = _mock_embedder(8)
        mock_llm.return_value = Mock()

        cfg = Config()
        cfg.memory.similarity_weight = 0.0
        cfg.memory.importance_weight = 0.0
        cfg.memory.recency_weight = 1.0
        cfg.memory.recency_half_life_seconds = 3600.0  # 1 hour

        cm = ContextManager(cfg)

        # Manually add two entries with different timestamps by overriding add_memory metadata
        now = time.time()
        cm.add_memory("older", {"t": 1})
        # Adjust timestamp to be older
        cm.long_term_memory.entries[-1].timestamp = now - 7200  # 2 hours ago

        cm.add_memory("newer", {"t": 2})
        cm.long_term_memory.entries[-1].timestamp = now - 60  # 1 min ago

        results = cm.query_memory("q", k=2)
        # Ensure 'newer' ranks higher
        assert results and 'newer' in results[0][0]


