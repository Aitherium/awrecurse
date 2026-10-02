"""Rank wide, read narrow: the ranker spends the iteration budget on the slices
the door believes in, teaches the door what each slice yielded, and degrades to
document order -- never to a failed read -- when the door is away."""

from __future__ import annotations

from awrecurse.engine import RecursionEngine
from awrecurse.ranker import DecideRanker, query_shape, slice_fingerprint


def _corpus(n=30, needle_at=25, width=2000):
    """n sections of exactly `width` chars, so chunk_size=width keeps each
    section in one slice (a needle split across a boundary is a different test)."""
    parts = []
    for i in range(n):
        head = ("Section %d. The launch code is 4471-ZETA. " % i) if i == needle_at else (
            "Section %d. Nothing relevant here about weather. " % i)
        parts.append((head + "filler " * 400)[:width].ljust(width))
    return "".join(parts)


def _complete(prompt: str) -> str:
    return "4471-ZETA" if "4471-ZETA" in prompt else "NOT_FOUND"


class FakeDoor:
    """A door that already LEARNED the needle slice (P(yes)=0.9) and knows nothing
    about the rest (source=none). Records outcomes."""

    def __init__(self, needle_word="launch"):
        self.needle_word, self.batches, self.outcomes, self.down = needle_word, 0, [], False

    def __call__(self, path, body):
        if self.down:
            raise OSError("refused")
        if path == "/decide/batch":
            self.batches += 1
            out = []
            for i, it in enumerate(body["items"]):
                # judge the SLICE half of the state; the query half names the needle too
                hit = self.needle_word in it["state"].split("|s:", 1)[1]
                out.append(
                    {
                        "decision_id": f"d{i}",
                        "answer": "yes" if hit else "no",
                        "confidence": 0.9 if hit else 0.0,
                        "source": "engine" if hit else "none",
                    }
                )
            return {"answers": out}
        if path == "/decide/outcome":
            self.outcomes.append((body["decision_id"], body["reward"]))
            return {"ok": True}
        raise AssertionError(path)


def test_document_order_misses_the_needle_within_the_budget():
    eng = RecursionEngine(_complete, chunk_size=2000, max_iterations=5)
    r = eng.recurse(_corpus(), "What is the launch code?")
    assert not r.success and r.iterations == 5


def test_ranker_reads_the_needle_first_and_teaches_the_door():
    door = FakeDoor()
    ranker = DecideRanker(post=door)
    eng = RecursionEngine(_complete, chunk_size=2000, max_iterations=5, ranker=ranker)
    r = eng.recurse(_corpus(), "What is the launch code?")
    assert r.success and r.final_answer == "4471-ZETA"
    assert r.iterations <= 5 and door.batches == 1, "one round trip ranked every slice"
    assert ranker.last_error is None
    # the slice that answered was rewarded, the ones that did not were penalised
    rewards = dict(door.outcomes)
    assert 1.0 in rewards.values() and all(v in (1.0, -1.0) for v in rewards.values())
    assert len(door.outcomes) == r.iterations


def test_door_down_degrades_to_document_order_and_says_so():
    door = FakeDoor()
    door.down = True
    ranker = DecideRanker(post=door)
    eng = RecursionEngine(_complete, chunk_size=2000, max_iterations=40, ranker=ranker)
    r = eng.recurse(_corpus(), "What is the launch code?")
    assert r.success, "a missing door must not cost the read"
    assert ranker.last_error and "document order" in ranker.last_error
    assert door.outcomes == []


def test_states_are_stable_across_filler_and_rechunking():
    assert query_shape("What is the launch code?") == query_shape("the LAUNCH code, what is it??")
    a = slice_fingerprint("alpha beta beta gamma " * 20)
    assert a == slice_fingerprint("alpha beta beta gamma " * 20)
    assert a != slice_fingerprint("alpha beta beta delta " * 20)


def test_a_door_that_answers_garbage_degrades_to_document_order():
    chunks = [(0, "alpha"), (5, "beta"), (10, "gamma")]
    for bad in ({"answers": [None, None, None]}, {"answers": [{"confidence": "high"}] * 3},
                ["not", "a", "dict"], {"answers": [{}]}):
        ranker = DecideRanker(post=lambda path, body, bad=bad: bad)
        assert ranker.rank("what is alpha?", chunks) == chunks
        assert ranker.last_error and "document order" in ranker.last_error
        ranker.outcome(0, True)  # nothing was ranked, so nothing is taught


def test_a_ranker_that_raises_is_recorded_and_the_read_still_happens():
    class Boom:
        def rank(self, query, chunks):
            raise RuntimeError("boom")

    eng = RecursionEngine(_complete, chunk_size=2000, max_iterations=40, ranker=Boom())
    r = eng.recurse(_corpus(), "What is the launch code?")
    assert r.success and "boom" in (eng.last_ranker_error or "")


def test_tls_verification_is_never_disabled(monkeypatch):
    import ssl
    import urllib.request

    seen = {}

    def fake_urlopen(req, timeout=None, context=None):
        seen["ctx"] = context
        raise OSError("stop here")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    ranker = DecideRanker(url="https://127.0.0.1:8197")
    assert ranker.rank("q about alpha", [(0, "alpha"), (5, "beta")]) == [(0, "alpha"), (5, "beta")]
    assert seen["ctx"].verify_mode == ssl.CERT_REQUIRED and seen["ctx"].check_hostname
