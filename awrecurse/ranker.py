"""Rank wide, read narrow.

`RecursionEngine.recurse` reads slices in document order until it hits
`max_iterations`. On a 200-slice context with a 10-iteration budget that is a
5% chance of reading the slice that holds the answer. A ranker decides the
ORDER, so the budget is spent on the slices most likely to answer.

`DecideRanker` asks the AitherOS decision door (`POST /decide/batch`, kind
`yesno`) "will this slice answer this question?" for every slice in ONE round
trip, reads in descending P(yes), and -- the part a static ranker cannot do --
posts the outcome of every slice it read (answered: +1, NOT_FOUND: -1). The
door learns per (question shape, slice fingerprint); the next question over the
same corpus is ranked from evidence, with no model call for slices it knows.

No hard dependency on anything but the stdlib: the door is plain JSON over HTTP
(AITHER_DECIDE_URL, default https://127.0.0.1:8197). TLS is always verified: a
door behind a private CA needs that CA in SSL_CERT_FILE (or REQUESTS_CA_BUNDLE).
When the door is unreachable, untrusted or answers garbage the ranker returns
document order and says so in `last_error`, so a missing door degrades to
today's behaviour rather than failing the read.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = frozenset(
    "the and for with that this from are was were you your have has not but its into "
    "than then them they their there these those which what when where who why how".split()
)

#: Everything a door can do wrong: unreachable, untrusted, not HTTP, not JSON, or
#: JSON of the wrong shape. All of it means "document order", never a failed read.
_DOOR_ERRORS = (
    urllib.error.URLError, http.client.HTTPException, OSError,
    ValueError, KeyError, TypeError, AttributeError,
)


def query_shape(query: str, k: int = 6) -> str:
    """The question reduced to its k rarest-looking content words, sorted, so the
    same question asked twice (or with different filler) maps to one state."""
    words = sorted({w.lower() for w in _WORD.findall(query) if w.lower() not in _STOP})
    words.sort(key=lambda w: (-len(w), w))
    return ",".join(words[:k])


def slice_fingerprint(text: str, k: int = 8) -> str:
    """Stable per-slice identity: the k most frequent content words plus a short
    content hash, so a slice re-chunked at the same boundary reads as itself."""
    freq: Dict[str, int] = {}
    for w in _WORD.findall(text):
        w = w.lower()
        if w not in _STOP:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq, key=lambda w: (-freq[w], w))[:k]
    h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]
    return ",".join(top) + "#" + h


class DecideRanker:
    """Order slices by the decision door's P(this slice answers the question)."""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        fork: str = "awrecurse.slice",
        timeout: float = 30.0,
        post: Optional[Callable[[str, dict], dict]] = None,
    ) -> None:
        self.url = (url or os.environ.get("AITHER_DECIDE_URL", "https://127.0.0.1:8197")).rstrip("/")
        self.token = (token or os.environ.get("AITHER_DECIDE_TOKEN")
                      or os.environ.get("AITHER_WM_INTERNAL_TOKEN"))
        self.fork = fork if fork.startswith("decide.") else f"decide.{fork}"
        self.timeout = timeout
        self._post = post or self._http_post
        self.last_error: Optional[str] = None
        self._pending: Dict[int, str] = {}  # slice start -> decision_id

    # -- transport ----------------------------------------------------------------
    def _http_post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-WM-Token"] = self.token
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        ctx = None
        if self.url.startswith("https"):
            ctx = ssl.create_default_context()
            bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
            # Verification is never switched off: the request carries a token.
            if bundle and os.path.isfile(bundle):
                ctx.load_verify_locations(bundle)
        with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as r:
            return json.loads(r.read().decode("utf-8") or "{}")

    # -- the two halves -------------------------------------------------------------
    def rank(self, query: str, chunks: Sequence[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """Return the chunks in the order they should be read. Document order on
        any failure (recorded in `last_error`), never an exception."""
        self.last_error = None
        self._pending = {}
        if len(chunks) < 2:
            return list(chunks)
        shape = query_shape(query)
        items = [
            {
                "domain": self.fork,
                "state": f"q:{shape}|s:{slice_fingerprint(text)}",
                "kind": "yesno",
                "question": "Does this slice contain the answer to the question?",
            }
            for _, text in chunks
        ]
        try:
            res = self._post("/decide/batch", {"items": items[:64]})
            answers = res.get("answers") or []
            if len(answers) != len(items[:64]):
                raise ValueError(f"door answered {len(answers)} of {len(items[:64])}")
            scored = []
            pending: Dict[int, str] = {}
            for (start, text), a in zip(chunks, answers):
                p_yes = float(a.get("confidence") or 0.0)
                if a.get("answer") == "no":
                    p_yes = 1.0 - p_yes
                if a.get("source") == "none":
                    p_yes = 0.5  # unknown, not unlikely
                pending[start] = str(a.get("decision_id") or "")
                scored.append((p_yes, start, text))
        except _DOOR_ERRORS as e:
            self.last_error = f"decide door unavailable ({e}); reading in document order"
            return list(chunks)
        self._pending = pending
        scored.sort(key=lambda t: (-t[0], t[1]))
        ordered = [(s, t) for _, s, t in scored]
        # anything past the batch cap keeps document order after the ranked head
        ordered.extend(chunks[64:])
        return ordered

    def outcome(self, start: int, answered: bool) -> None:
        """Teach the door what happened to a slice it ranked."""
        did = self._pending.pop(start, "")
        if not did:
            return
        try:
            self._post("/decide/outcome", {"decision_id": did, "reward": 1.0 if answered else -1.0})
        except _DOOR_ERRORS as e:
            self.last_error = f"outcome not delivered ({e})"


__all__ = ["DecideRanker", "query_shape", "slice_fingerprint"]
