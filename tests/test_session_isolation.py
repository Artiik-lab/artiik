from unittest.mock import Mock, patch
import numpy as np

from context_manager.core import ContextManager


def _mock_embedder(dim=16):
    class M:
        def get_sentence_embedding_dimension(self):
            return dim

        def encode(self, texts, convert_to_numpy=True):
            import numpy as _np
            n = len(texts) if isinstance(texts, list) else 1
            return _np.ones((n, dim), dtype=_np.float32)

    return M()


def test_session_scoping_filters_results():
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls, \
         patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
        mock_model_cls.return_value = _mock_embedder(16)
        mock_llm.return_value = Mock()

        # Create two managers with different sessions
        cm_a = ContextManager(session_id="A")
        cm_b = ContextManager(session_id="B")

        # Add memories into both sessions
        cm_a.add_memory("alpha")
        cm_b.add_memory("beta")

        # Query with isolation (default deny cross-session)
        results_a = cm_a.query_memory("a", k=10)
        assert any("alpha" in t for t, _ in results_a)
        assert not any("beta" in t for t, _ in results_a)

        # Allow cross-session for B and query
        cm_b.set_session("B", allow_cross_session=True)
        results_b = cm_b.query_memory("b", k=10)
        # With trivial embeddings, both may show up; ensure at least beta is present
        assert any("beta" in t for t, _ in results_b)


