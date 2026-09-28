"""Named regression tests for the defects found in artiik 0.1.

Each test states a 0.1 defect and the behavior that must hold instead. They're
skipped until the feature they cover lands, and the skip reason names the issue
that brings it. The 0.1 code is at the ``v0.1.1`` tag.
"""

import pytest

COMPACTION = "needs compaction (#19)"
MEMORY = "needs scoped memory recall (#21)"


def pending() -> None:
    pytest.fail("write this regression test together with the feature it covers")


@pytest.mark.skip(reason=COMPACTION)
def test_v01_the_summary_path_runs() -> None:
    """0.1 evicted old turns to stay under the short-term capacity before it
    checked whether that capacity was exceeded, so it never summarized and
    dropped old turns silently (``ContextManager.observe``).

    A session that outgrows its budget must compact, and every turn that
    leaves the context must be covered by a summary.
    """
    pending()


@pytest.mark.skip(reason=MEMORY)
def test_v01_no_duplicate_hits_when_k_exceeds_the_memory_count() -> None:
    """0.1 read FAISS's ``-1`` padding as a list index when ``k`` was larger
    than the number of memories, so the last memory came back several times
    (``LongTermMemory.search``).

    A lookup returns each memory at most once, and at most as many hits as
    there are visible memories.
    """
    pending()


@pytest.mark.skip(reason=MEMORY)
def test_v01_deleting_after_a_load_keeps_the_other_memories() -> None:
    """0.1 loaded entries with zero vectors as placeholders, and a delete
    rebuilt the index from them, so every memory became unreachable
    (``LongTermMemory.load`` then ``delete_memory``).

    After a save, a load and a delete, every other memory is still found by
    the lookups that found it before.
    """
    pending()


@pytest.mark.skip(reason=MEMORY)
def test_v01_scope_filtering_happens_before_ranking() -> None:
    """0.1 took the top ``k`` hits across all scopes and filtered them by
    scope afterwards, so a scope whose memories ranked lower got nothing
    (``ContextManager.build_context``).

    A lookup ranks only the memories visible in its scope: with many better
    matches in other scopes, it still returns the scope's own memories.
    """
    pending()


@pytest.mark.skip(reason=MEMORY)
def test_v01_every_lookup_path_applies_the_scope() -> None:
    """0.1 applied scopes in ``build_context`` but not in ``query_memory``,
    which returned memories from every session and task.

    Every way to read memories (automatic recall, direct queries, the memory
    tool) passes :func:`artiik.testing.check_scopes`.
    """
    pending()
