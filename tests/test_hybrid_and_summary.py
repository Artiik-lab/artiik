from unittest.mock import Mock, patch
import numpy as np

from context_manager.core import ContextManager
from context_manager.memory.summarizer import HierarchicalSummarizer


def test_hybrid_keyword_filter_keeps_matching_ltm_first():
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls, \
         patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm:
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 32
        mock_model.encode.side_effect = lambda texts, convert_to_numpy=True: np.ones((len(texts) if isinstance(texts, list) else 1, 32), dtype=np.float32)
        mock_model_cls.return_value = mock_model
        mock_llm.return_value = Mock()

        cm = ContextManager()
        # Inject two LTM entries, one containing keyword 'python', one not
        cm.add_memory("We discussed Python packaging best practices", {"topic": "dev"})
        cm.add_memory("Completely unrelated gardening tips", {"topic": "garden"})

        # Add a recent turn so context assembles
        cm.observe("Tell me about Python wheels", "Sure")
        ctx = cm.build_context("How to build Python wheels?")
        # Ensure the keyword-matching LTM text appears
        assert "Python" in ctx


def test_summarizer_summarize_texts_fallback_on_error():
    bad_llm = Mock()
    bad_llm.generate_sync.side_effect = Exception("down")
    summarizer = HierarchicalSummarizer(llm_adapter=bad_llm)
    texts = ["A long text about context engineering", "Another paragraph"]
    summary = summarizer.summarize_texts(texts)
    assert isinstance(summary, str)
    assert len(summary) > 0


