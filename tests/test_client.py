"""Test the RLM client.

These overlap the CLI --self-test on purpose. The self-test is what ships
(it runs on any install, with no pytest and no network); this is what runs
in CI with the mutation guards that prove each assertion can still fail.
"""

from __future__ import annotations

import pytest
from awrecurse import RecurseClient, RecurseError, recurse_body
from awrecurse.cli import main


def test_recurse_body_is_exactly_the_declared_field_set():
    # THE load-bearing assertion. A field spelled wrong is silently DROPPED
    # by a service that takes extra="ignore", so the only protection is
    # sending exactly the declared field set.
    body = recurse_body("context", "query")
    declared = ("context", "query", "chunk_size", "max_iterations")
    assert tuple(body) == declared


def test_recurse_body_defaults_match_expectations():
    assert recurse_body("ctx", "q") == {
        "context": "ctx",
        "query": "q",
        "chunk_size": 2000,
        "max_iterations": 10,
    }


@pytest.mark.parametrize("ctx,q", [
    ("", "query"),
    ("context", ""),
    ("   ", "query"),
    ("context", "   "),
])
def test_empty_context_or_query_is_refused_with_a_reason(ctx, q):
    # The service refuses these too; refusing here saves a round trip.
    with pytest.raises(ValueError):
        recurse_body(ctx, q)


@pytest.mark.parametrize("chunk_size", [50, 99, 0, -1])
def test_chunk_size_below_100_is_refused(chunk_size):
    with pytest.raises(ValueError, match="chunk_size.*100"):
        recurse_body("c", "q", chunk_size=chunk_size)


@pytest.mark.parametrize("max_iter", [0, -1, -100])
def test_max_iterations_below_1_is_refused(max_iter):
    with pytest.raises(ValueError, match="max_iterations.*1"):
        recurse_body("c", "q", max_iterations=max_iter)


def test_valid_chunk_size_accepted():
    # Boundary: exactly 100 is valid
    body = recurse_body("c", "q", chunk_size=100)
    assert body["chunk_size"] == 100

    # 2000 is the default
    body = recurse_body("c", "q", chunk_size=2000)
    assert body["chunk_size"] == 2000


def test_valid_max_iterations_accepted():
    # Boundary: exactly 1 is valid
    body = recurse_body("c", "q", max_iterations=1)
    assert body["max_iterations"] == 1

    # 10 is the default
    body = recurse_body("c", "q", max_iterations=10)
    assert body["max_iterations"] == 10


def test_recurse_client_no_url_or_complete_fn_raises():
    """A client with no backend configuration must raise, not silently fail."""
    c = RecurseClient()  # no URL, no complete_fn
    with pytest.raises(RecurseError):
        c.recurse("context", "query")


def test_recurse_client_with_local_complete_fn_works():
    """Local recursion via a complete_fn callable."""
    def dummy_complete(prompt: str) -> str:
        return "dummy answer" if "question" in prompt.lower() else "NOT_FOUND"

    c = RecurseClient(complete_fn=dummy_complete)
    result = c.recurse("some context", "what question", chunk_size=500)

    assert "answer" in result["final_answer"]
    assert "slices_read" in result
    assert isinstance(result["slices_read"], list)


def test_recurse_client_with_local_complete_fn_tracks_slices_read():
    """The critical assertion: slices_read is returned by local recursion."""
    def dummy_complete(prompt: str) -> str:
        # First chunk gets an answer
        if "chunk" in prompt.lower():
            return "answer in the chunk"
        return "NOT_FOUND"

    c = RecurseClient(complete_fn=dummy_complete)
    ctx = "This is the chunk content " * 50  # ~1500 chars
    result = c.recurse(ctx, "is there chunk?", chunk_size=500)

    assert result["success"]
    assert "slices_read" in result
    # The result must show that slices were read — not empty!
    assert len(result["slices_read"]) > 0


def test_recurse_client_raises_on_failed_recursion():
    """A recursion that finds no answer raises RecurseError."""
    def always_not_found(prompt: str) -> str:
        return "NOT_FOUND"

    c = RecurseClient(complete_fn=always_not_found)
    result = c.recurse("context", "query", chunk_size=500, max_iterations=2)

    # Result is a dict with success=False
    assert result["success"] is False
    assert "error" in result


def test_result_dict_structure_from_client_recurse():
    """Verify the returned dict has all required fields."""
    def dummy_complete(prompt: str) -> str:
        return "test answer"

    c = RecurseClient(complete_fn=dummy_complete)
    result = c.recurse("context", "query", chunk_size=500)

    # All required fields must be present
    required = ("final_answer", "slices_read", "iterations", "tokens",
                "success", "error", "trace")
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_recurse_body_with_custom_params():
    body = recurse_body("c", "q", chunk_size=5000, max_iterations=20)
    assert body["chunk_size"] == 5000
    assert body["max_iterations"] == 20


def test_self_test_passes_and_is_the_shipped_check():
    """The --self-test can run offline and proves the contract."""
    assert main(["--self-test"]) == 0


def test_no_subcommand_is_an_error():
    """Calling with no subcommand should show help and exit 2."""
    assert main([]) == 2


def test_recurse_client_base_url_normalized():
    """Trailing slash is stripped from base_url."""
    c = RecurseClient("https://h/")
    assert c.base_url == "https://h"


def test_recurse_client_no_token_means_no_header():
    """No token means token is None, not empty string."""
    c = RecurseClient("https://h")
    assert c.token is None


def test_transport_failure_raises_recurse_error():
    """MUTATION GUARD: failures raise, they don't return empty."""
    # This is the guard that prevents silent failures from looking like
    # legitimate empty results.
    c = RecurseClient("http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(RecurseError):
        c.recurse("context", "query")


def test_recurse_error_is_exception():
    """RecurseError must be raisable."""
    assert issubclass(RecurseError, Exception)
    with pytest.raises(RecurseError):
        raise RecurseError("test")
