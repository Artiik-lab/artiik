import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np

from context_manager.core import ContextManager


def test_embedding_dimension_validation_and_alignment():
    # Mock embedding provider to return fixed-dimension vectors
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls:
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        # Encode returns a numpy array of shape (n, 384)
        def fake_encode(texts, convert_to_numpy=True):
            if isinstance(texts, list):
                return np.ones((len(texts), 384), dtype=np.float32)
            return np.ones((1, 384), dtype=np.float32)
        mock_model.encode.side_effect = fake_encode
        mock_model_cls.return_value = mock_model

        cm = ContextManager()  # Should auto-align to 384
        # Ensure dimension is aligned
        assert cm.long_term_memory.dimension == 384

        # Add memory should not raise due to dimension mismatch
        cm.add_memory("test entry")
        assert len(cm.long_term_memory) == 1


def test_cosine_similarity_proxy_monotonicity():
    # Create two embeddings with different cosine similarity to a query
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls:
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        # Define deterministic encodings
        vectors = {
            'query': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'a': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'b': np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

        def fake_encode(texts, convert_to_numpy=True):
            arr = []
            if isinstance(texts, list):
                for t in texts:
                    v = vectors.get(t, vectors['a'])
                    arr.append(v)
            else:
                arr.append(vectors.get(texts, vectors['a']))
            return np.stack(arr, axis=0)

        mock_model.encode.side_effect = fake_encode
        mock_model_cls.return_value = mock_model

        cm = ContextManager()
        # Add two memories with different angles to the query
        cm.add_memory('a')  # identical to query -> highest similarity
        cm.add_memory('b')  # orthogonal -> lower similarity

        results = cm.query_memory('query', k=2)
        # First result should be 'a'
        assert any('a' in text for text, _ in results[:1])


def test_budget_aware_pruning_keeps_within_budget():
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls:
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = lambda texts, convert_to_numpy=True: np.ones((len(texts) if isinstance(texts, list) else 1, 384), dtype=np.float32)
        mock_model_cls.return_value = mock_model

        cm = ContextManager()
        # Set a very small budget to force pruning
        cm.config.memory.prompt_token_budget = 50

        # Populate STM with verbose turns
        for i in range(5):
            cm.observe("User " + ("x" * 50), "Assistant " + ("y" * 50))

        # Build context should fit budget
        context = cm.build_context("Q")
        assert cm.token_counter.count_tokens(context) <= cm.config.memory.prompt_token_budget


def test_ltm_persistence_save_and_load():
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls:
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 8
        mock_model.encode.side_effect = lambda texts, convert_to_numpy=True: np.ones((len(texts) if isinstance(texts, list) else 1, 8), dtype=np.float32)
        mock_model_cls.return_value = mock_model

        cm = ContextManager()
        cm.add_memory("alpha", {"t": 1})
        cm.add_memory("beta", {"t": 2})

        tmpdir = tempfile.mkdtemp(prefix="cm_ltm_")
        try:
            cm.save_memory(tmpdir)

            # New instance, load
            cm2 = ContextManager()
            cm2.load_memory(tmpdir)

            stats = cm2.get_stats()
            assert stats['long_term_memory']['num_entries'] >= 2
        finally:
            shutil.rmtree(tmpdir)


