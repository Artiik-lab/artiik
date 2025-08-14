import time
from unittest.mock import Mock, patch

from context_manager.memory.summarizer import HierarchicalSummarizer
from context_manager.memory.short_term import Turn
from context_manager.core import ContextManager
from context_manager.utils.token_counter import TokenCounter


def test_summarizer_fallback_on_llm_error():
    bad_adapter = Mock()
    bad_adapter.generate_sync.side_effect = Exception("LLM down")

    summarizer = HierarchicalSummarizer(llm_adapter=bad_adapter)

    turns = [
        Turn(user_input="Hello", assistant_response="Hi", token_count=5, timestamp=time.time()),
        Turn(user_input="Plan trip", assistant_response="Sure", token_count=7, timestamp=time.time()),
    ]

    summary = summarizer.summarize_chunk(turns)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_summary_metadata_fields():
    summarizer = HierarchicalSummarizer(llm_adapter=Mock())
    turns = [
        Turn(user_input="A", assistant_response="B", token_count=3, timestamp=time.time()),
        Turn(user_input="C", assistant_response="D", token_count=4, timestamp=time.time()),
    ]
    summary = "Summary text"
    meta = summarizer.create_summary_metadata(turns, summary)
    assert meta["num_turns"] == 2
    assert "summary_length" in meta and meta["summary_length"] == len(summary)
    assert "compression_ratio" in meta


def test_debug_context_building_counts_and_budget():
    with patch('context_manager.llm.embeddings.SentenceTransformer') as mock_model_cls, \
         patch('context_manager.llm.adapters.create_llm_adapter') as mock_llm_factory:
        # Mock embedder
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = lambda texts, convert_to_numpy=True: __import__('numpy').ones((len(texts) if isinstance(texts, list) else 1, 384), dtype='float32')
        mock_model_cls.return_value = mock_model
        # Mock LLM adapter (not called in debug)
        mock_llm_factory.return_value = Mock()

        cm = ContextManager()
        cm.config.memory.prompt_token_budget = 200

        cm.observe("Hello", "Hi")
        cm.observe("How are you?", "Fine")

        dbg = cm.debug_context_building("What did we discuss?")
        assert dbg["recent_turns_count"] >= 1
        assert dbg["final_context_tokens"] <= dbg["context_budget"]


def test_tokencounter_fallback_on_encoder_error():
    counter = TokenCounter()

    class BrokenEncoder:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("fail")

    # Inject broken encoder to force fallback path
    counter._encoder = BrokenEncoder()
    text = "abcd" * 10  # length 40, expect ~10 tokens fallback
    tokens = counter.count_tokens(text)
    assert tokens == len(text) // 4


