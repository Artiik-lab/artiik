from unittest.mock import Mock, patch


def _mock_embedder(dim=16):
    class M:
        def get_sentence_embedding_dimension(self):
            return dim

        def encode(self, texts, convert_to_numpy=True):
            import numpy as _np
            n = len(texts) if isinstance(texts, list) else 1
            return _np.ones((n, dim), dtype=_np.float32)

    return M()


def test_task_scoping_filters_results():
    from context_manager.core import ContextManager

    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls, \
         patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
        mock_model_cls.return_value = _mock_embedder(16)
        mock_llm.return_value = Mock()

        # Two managers for two tasks
        cm_x = ContextManager(task_id="X")
        cm_y = ContextManager(task_id="Y")

        # Add memories scoped via orchestrator (it injects task_id into metadata)
        cm_x.add_memory("task X note")
        cm_y.add_memory("task Y note")

        # Query with isolation (default deny cross-task)
        results_x = cm_x.query_memory("note", k=10)
        assert any("task X note" in t for t, _ in results_x)
        assert not any("task Y note" in t for t, _ in results_x)

        # Allow cross-task for Y and query
        cm_y.set_task("Y", allow_cross_task=True)
        results_y = cm_y.query_memory("note", k=10)
        # Ensure at least the Y-scoped entry is present
        assert any("task Y note" in t for t, _ in results_y)


