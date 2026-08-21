"""Test the RLM engine — the core recursion logic.

What a stranger's machine must still be true of awrecurse.
"""

from __future__ import annotations

from awrecurse.engine import RecursionEngine, RLMResult


class TestChunking:
    """Verify that context is split correctly."""

    def test_small_context_is_one_chunk(self):
        engine = RecursionEngine(chunk_size=2000)
        chunks = engine.chunk_context("hello")
        assert len(chunks) == 1
        assert chunks[0] == (0, "hello")

    def test_context_exactly_chunk_size_is_one_chunk(self):
        engine = RecursionEngine(chunk_size=2000)
        ctx = "x" * 2000
        chunks = engine.chunk_context(ctx)
        assert len(chunks) == 1
        assert chunks[0] == (0, ctx)

    def test_context_one_more_than_chunk_size_is_two_chunks(self):
        engine = RecursionEngine(chunk_size=2000)
        ctx = "x" * 2001
        chunks = engine.chunk_context(ctx)
        assert len(chunks) == 2
        assert chunks[0] == (0, "x" * 2000)
        assert chunks[1] == (2000, "x")

    def test_large_context_splits_into_many_chunks(self):
        engine = RecursionEngine(chunk_size=2000)
        ctx = "x" * 5000
        chunks = engine.chunk_context(ctx)
        assert len(chunks) == 3
        assert chunks[0][1] == "x" * 2000
        assert chunks[1][1] == "x" * 2000
        assert chunks[2][1] == "x" * 1000
        # Verify start positions
        assert chunks[0][0] == 0
        assert chunks[1][0] == 2000
        assert chunks[2][0] == 4000

    def test_empty_context_returns_one_empty_chunk(self):
        engine = RecursionEngine()
        chunks = engine.chunk_context("")
        assert len(chunks) == 1
        assert chunks[0] == (0, "")


class TestQueryChunk:
    """Verify single-chunk queries."""

    def test_query_chunk_with_answer_returns_answer(self):
        def complete(prompt: str) -> str:
            return "yes, the answer is 42"

        engine = RecursionEngine(complete_fn=complete)
        answer = engine.query_chunk("context", "what is the answer?")
        assert answer == "yes, the answer is 42"

    def test_query_chunk_not_found_returns_empty(self):
        def complete(prompt: str) -> str:
            return "NOT_FOUND"

        engine = RecursionEngine(complete_fn=complete)
        answer = engine.query_chunk("context", "what is the answer?")
        assert answer == ""

    def test_query_chunk_case_insensitive_not_found(self):
        for response in ("NOT_FOUND", "not_found", "[NOT_FOUND]", "NOT FOUND"):
            def complete(prompt: str) -> str:
                return response

            engine = RecursionEngine(complete_fn=complete)
            answer = engine.query_chunk("context", "what?")
            assert answer == "", f"Failed for response: {response}"


class TestRecursion:
    """Test the core recursive algorithm."""

    def test_recurse_empty_context_fails(self):
        def complete(prompt: str) -> str:
            return "answer"

        engine = RecursionEngine(complete_fn=complete)
        result = engine.recurse("", "query")
        assert not result.success
        assert result.error == "Context is empty"
        assert result.iterations == 0

    def test_recurse_single_chunk_with_answer(self):
        def complete(prompt: str) -> str:
            return "the answer is here"

        engine = RecursionEngine(complete_fn=complete, chunk_size=2000)
        ctx = "a" * 500
        result = engine.recurse(ctx, "what?")

        assert result.success
        assert result.final_answer == "the answer is here"
        assert result.iterations == 1
        assert len(result.slices_read) == 1
        assert result.slices_read[0] == (0, 500)

    def test_recurse_multi_chunk_finds_answer_in_first(self):
        """If first chunk has the answer, continue checking others too."""
        call_count = [0]

        def complete(prompt: str) -> str:
            call_count[0] += 1
            # Only first chunk has an answer
            return "answer in first chunk" if call_count[0] == 1 else "NOT_FOUND"

        engine = RecursionEngine(complete_fn=complete, chunk_size=1000)
        ctx = "a" * 3000
        result = engine.recurse(ctx, "what?")

        assert result.success
        assert "answer" in result.final_answer
        # First chunk is included in slices_read
        assert len(result.slices_read) >= 1

    def test_recurse_multi_chunk_finds_answer_in_middle(self):
        """If middle chunk has the answer, it should be found."""
        responses = ["NOT_FOUND", "the middle chunk has it", "NOT_FOUND"]
        idx = [0]

        def complete(prompt: str) -> str:
            resp = responses[idx[0]]
            idx[0] += 1
            return resp

        engine = RecursionEngine(complete_fn=complete, chunk_size=1000,
                               max_iterations=5)
        ctx = "a" * 3000
        result = engine.recurse(ctx, "what?")

        assert result.success
        assert result.final_answer == "the middle chunk has it"
        # Should have checked first, then middle, then aggregated
        assert result.iterations >= 1

    def test_slices_read_tracks_which_parts_were_examined(self):
        """The critical test: slices_read must show which parts were read."""
        # This is the MUTATION GUARD that proves the whole package.
        # Without slices_read, silent truncation is invisible.
        responses = ["NOT_FOUND", "answer from second chunk", "NOT_FOUND"]
        idx = [0]

        def complete(prompt: str) -> str:
            resp = responses[idx[0]]
            idx[0] += 1
            return resp

        engine = RecursionEngine(complete_fn=complete, chunk_size=2000,
                               max_iterations=5)
        ctx = "x" * 6000  # 3 chunks
        result = engine.recurse(ctx, "what?")

        # The answer came from the middle chunk
        assert result.final_answer == "answer from second chunk"
        # slices_read MUST show that the middle chunk was examined
        # (start at 2000, end at 4000 for a 2000-char chunk)
        assert (2000, 4000) in result.slices_read

        # This is not just "it returned something" — it's "it read the middle".
        # A broken implementation might return the answer but claim to have
        # read only the first chunk, which is the exact bug this package
        # exists to prevent.

    def test_iteration_cap_prevents_runaway(self):
        """Max iterations caps the recursion depth."""
        call_count = [0]

        def complete(prompt: str) -> str:
            call_count[0] += 1
            return "NOT_FOUND"

        engine = RecursionEngine(complete_fn=complete, chunk_size=1000,
                               max_iterations=3)
        ctx = "x" * 10000  # 10 chunks
        result = engine.recurse(ctx, "what?")

        # Should have stopped at or before max_iterations (may hit synthesis too)
        assert result.iterations <= engine.max_iterations + 1
        assert not result.success

    def test_no_answer_returns_failure(self):
        """If no chunk has an answer, return failure with explanation."""
        def complete(prompt: str) -> str:
            return "NOT_FOUND"

        engine = RecursionEngine(complete_fn=complete, chunk_size=1000,
                               max_iterations=5)
        ctx = "x" * 3000
        result = engine.recurse(ctx, "what?")

        assert not result.success
        assert result.final_answer == ""
        assert "No answer found" in result.error
        assert len(result.slices_read) == 0


class TestRLMResult:
    """Test the result object."""

    def test_result_to_dict_has_all_fields(self):
        result = RLMResult(
            final_answer="test answer",
            slices_read=[(0, 100), (200, 300)],
            iterations=2,
            tokens=150,
            success=True,
        )
        d = result.to_dict()
        assert d["final_answer"] == "test answer"
        assert d["slices_read"] == [(0, 100), (200, 300)]
        assert d["iterations"] == 2
        assert d["tokens"] == 150
        assert d["success"] is True

    def test_result_defaults(self):
        result = RLMResult()
        assert result.final_answer == ""
        assert result.slices_read == []
        assert result.iterations == 0
        assert result.tokens == 0
        assert result.success is True
        assert result.error == ""


class TestSilentTruncationDetection:
    """The raison d'être: detect silent truncation."""

    def test_answer_from_middle_proves_slices_read_is_real(self):
        """
        MUTATION GUARD: This test MUST fail if slices_read is hard-coded
        or if the engine queries only the ends.

        The bug: a model given a 100k-word document with a 4k word window
        drops 96k words from the middle, answers fluently from the ends,
        and you never know the middle was missing.

        This test simulates that: put a crucial answer in the MIDDLE chunk,
        surrounded by empty chunks, and verify that slices_read INCLUDES
        the middle.
        """
        def complete(prompt: str) -> str:
            # Simulate: only the middle chunk has content
            if "THE REAL ANSWER IS HERE" in prompt:
                return "THE REAL ANSWER IS HERE in the middle"
            return "NOT_FOUND"

        engine = RecursionEngine(complete_fn=complete, chunk_size=2000,
                               max_iterations=10)

        # Build a context: empty | THE ANSWER | empty
        ctx = (
            "padding " * 250 +  # ~2000 chars, chunk 0
            "THE REAL ANSWER IS HERE " * 80 +  # ~2000 chars, chunk 1
            "padding " * 250  # ~2000 chars, chunk 2
        )

        result = engine.recurse(ctx, "where is the answer?")

        # The answer was found
        assert result.success
        assert "REAL ANSWER" in result.final_answer

        # And slices_read MUST include the middle chunk
        # (which is approximately at position 2000 to 4000)
        middle_chunk_touched = any(
            start >= 1900 and end <= 4100
            for start, end in result.slices_read
        )
        assert middle_chunk_touched, (
            f"Middle chunk was not read! slices_read = {result.slices_read}. "
            "This is the bug the package is written to prevent."
        )
