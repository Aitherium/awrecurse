"""Client for an RLM-shaped recursion service.

WHY A CLIENT AND NOT A LIFT
===========================
The monorepo's RLM implementation is a complex cognitive system with 2,500+ lines
and dozens of platform imports. Lifting it into a package produces something that
raises `ModuleNotFoundError` on a stranger's machine while reading as authoritative
— a broken package, not a shipped one.

Instead, ship the small standalone CLIENT. The recursion algorithm is the product,
and a local engine is good enough for offline use. Connect to a real service when
you have one.

WHAT "SHAPED" MEANS
===================
Any service exposing these routes — read off the running service, not invented:

    POST /recurse          {context, query, chunk_size, max_iterations}
    GET  /health           health check
    GET  /config           service configuration

THE TRAP THIS CLIENT EXISTS TO AVOID
====================================
An overflow-ed context returns fluent, wrong answers with no error. A test
asserting "returns a string" passes on that bug. The ONLY protection is tracking
which slices were actually examined, so a result's `slices_read` is not optional
— it is the whole point.

Depends on httpx and nothing else.
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = ["RecurseClient", "RecurseError"]

DEFAULT_TIMEOUT = 60.0

#: Standard result keys the service declares.
RESULT_FIELDS = ("final_answer", "slices_read", "iterations", "tokens",
                 "success", "error")


class RecurseError(RuntimeError):
    """The service refused or could not answer.

    Raised, never returned as an empty result. A recursion that failed and one
    that genuinely could not find an answer are different facts, and a client
    that returns empty for both makes a dead backend look like an empty result
    — the exact silence that hides an outage.
    """


def recurse_body(context: str, query: str, *, chunk_size: int = 2000,
                 max_iterations: int = 10) -> dict:
    """The request body for POST /recurse.

    Raises ValueError for the things the service will also reject, so the caller
    gets the real reason rather than an empty result.
    """
    c = (context or "").strip()
    q = (query or "").strip()

    if not c:
        raise ValueError("context must not be empty")
    if not q:
        raise ValueError("query must not be empty")
    if chunk_size < 100:
        raise ValueError("chunk_size must be >= 100 (smaller chunks produce "
                        "more calls than answers)")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    return {
        "context": c,
        "query": q,
        "chunk_size": chunk_size,
        "max_iterations": max_iterations,
    }


class RecurseClient:
    """Talks to an RLM-shaped recursion service."""

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None, *,
                 timeout: float = DEFAULT_TIMEOUT,
                 complete_fn: Optional[Callable[[str], str]] = None,
                 verify: bool | str = True) -> None:
        """
        base_url     the service origin. If None, uses complete_fn (local mode).
        token        the CALLER's bearer, never a service credential. This package
                     ships publicly, so an internal key would either fail for
                     strangers or work for everyone who reads the source.
        timeout      request timeout in seconds.
        complete_fn  a local completion function (sync). If provided, recursion
                     runs locally without needing a service.
        verify       never False against a real deployment; trust the CA instead.
        """
        self.base_url = base_url.rstrip("/") if base_url else None
        self.token = token
        self.timeout = timeout
        self.complete_fn = complete_fn
        self.verify = verify

    def _http(self):
        import httpx  # local import so the module imports without httpx present

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        return httpx.Client(base_url=self.base_url, headers=headers,
                           timeout=self.timeout, verify=self.verify)

    def _post(self, path: str, body: dict) -> dict:
        if not self.base_url:
            raise RecurseError("No service URL configured; use complete_fn for "
                             "local recursion, or pass base_url to __init__")
        try:
            with self._http() as c:
                r = c.post(path, json=body)
        except Exception as exc:  # noqa: BLE001
            raise RecurseError(f"{path}: {exc}") from exc
        if r.status_code >= 400:
            raise RecurseError(f"{path}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _get(self, path: str) -> dict:
        if not self.base_url:
            raise RecurseError("No service URL configured")
        try:
            with self._http() as c:
                r = c.get(path)
        except Exception as exc:  # noqa: BLE001
            raise RecurseError(f"{path}: {exc}") from exc
        if r.status_code >= 400:
            raise RecurseError(f"{path}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    # ── Recursion ───────────────────────────────────────────────────────────

    def recurse(self, context: str, query: str, *, chunk_size: int = 2000,
                max_iterations: int = 10) -> dict:
        """Recursively query a context larger than the model's window.

        If a local complete_fn was provided, uses local recursion.
        Otherwise, delegates to the service.

        Returns a dict with:
            final_answer: The model's answer.
            slices_read: List of (start, end) tuples showing which parts were read.
            iterations: How many recursive calls it took.
            tokens: Token count (if available).
            success: Whether an answer was found.
            error: Error message (if success is False).
        """
        body = recurse_body(context, query, chunk_size=chunk_size,
                           max_iterations=max_iterations)

        if self.complete_fn:
            # Local recursion
            from awrecurse.engine import RecursionEngine
            engine = RecursionEngine(self.complete_fn, chunk_size=chunk_size,
                                     max_iterations=max_iterations)
            result = engine.recurse(context, query)
            return result.to_dict()

        # Remote recursion
        return self._post("/recurse", body)

    # ── About the service ───────────────────────────────────────────────────

    def health(self) -> dict:
        """Health check. Returns service status."""
        return self._get("/health")

    def config(self) -> dict:
        """Service configuration and capabilities."""
        return self._get("/config")
