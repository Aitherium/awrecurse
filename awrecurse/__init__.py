"""awrecurse — Recursive Language Model client.

Answer a question over a context far larger than the model's window — recursively,
with the trace of which slices were actually read kept intact.

The problem: a context that overflows does not raise. The middle is dropped, the
model answers fluently from the ends, and the reply looks exactly like one drawn
from the whole document. The failure is a SILENCE.

The solution: split an oversized context into slices, query each recursively,
aggregate results, and track which parts were examined.

    from awrecurse import RecurseClient

    c = RecurseClient("https://recurse.example.com", token="...")
    result = c.recurse(huge_document, "what changed in version 2.1?")
    print(result["final_answer"])
    print(result["slices_read"])  # which parts the model actually examined

Read `client.py` and `engine.py` before using the service: the contract is that
every result includes slices_read — without it, the failure case (silent
truncation) is invisible.
"""

from __future__ import annotations

from awrecurse.client import (
    RecurseClient,
    RecurseError,
    recurse_body,
)
from awrecurse.engine import (
    RecursionEngine,
    RLMResult,
)

__version__ = "0.1.0"

__all__ = [
    "RecurseClient",
    "RecurseError",
    "RecursionEngine",
    "RLMResult",
    "recurse_body",
    "__version__",
]
