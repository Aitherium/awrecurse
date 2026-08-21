"""Recursive Language Model engine — the core RLM algorithm.

The problem this solves:
    A context window that overflows does not raise. The middle is dropped silently,
    the model answers fluently from the ends, and the reply looks exactly like one
    drawn from the whole document. The failure is a SILENCE.

The solution:
    Split an oversized context into slices, query each recursively, aggregate
    results, and keep the trace of which slices were actually read. A caller can
    see exactly which parts the model examined.

Algorithm (from Zhang, Kraska & Khattab, arXiv:2512.24601):
    1. Chunk context into slices smaller than the window.
    2. Query the first slice for a direct answer. If found, return it and the
       slice index.
    3. If not found, recursively query the REMAINING slices by aggregating their
       answers.
    4. Cap iterations to prevent runaway recursion; return the best answer found.

Depends on httpx and nothing else. Works with any completion-shaped endpoint
(OpenAI-compatible POST /v1/chat/completions is the default).
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = ["RecursionEngine", "RLMResult"]

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_MAX_ITERATIONS = 10


class RLMResult:
    """The result of an RLM completion.

    Attributes:
        final_answer: The answer the model converged on.
        slices_read: List of (start_char, end_char) tuples showing which parts
                     of the context were actually examined.
        iterations: How many recursive calls it took.
        tokens: Optional token count (if the backend reported it).
        success: Whether a real answer was found vs. max_iterations exceeded.
        error: An error message if success is False.
    """

    __slots__ = ("final_answer", "slices_read", "iterations", "tokens",
                 "success", "error", "raw_trace")

    def __init__(self, final_answer: str = "", slices_read: Optional[list] = None,
                 iterations: int = 0, tokens: int = 0, success: bool = True,
                 error: str = "", raw_trace: Optional[dict] = None) -> None:
        self.final_answer = final_answer
        self.slices_read = slices_read or []
        self.iterations = iterations
        self.tokens = tokens
        self.success = success
        self.error = error
        self.raw_trace = raw_trace or {}

    def __repr__(self) -> str:
        return (f"<RLMResult success={self.success} iterations={self.iterations} "
                f"slices_read={len(self.slices_read)}>"
               )

    def to_dict(self) -> dict:
        """Return the full trace as a dict."""
        return {
            "final_answer": self.final_answer,
            "slices_read": self.slices_read,
            "iterations": self.iterations,
            "tokens": self.tokens,
            "success": self.success,
            "error": self.error,
            "trace": self.raw_trace,
        }


class RecursionEngine:
    """Recursive Language Model engine.

    Works with any completion-shaped backend (any callable that takes a prompt
    and returns a string, or any OpenAI-compatible endpoint).
    """

    def __init__(self, complete_fn: Optional[Callable[[str], str]] = None, *,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS) -> None:
        """
        complete_fn: A callable(prompt: str) -> str that completes a prompt.
                     If None, you must pass backend_url and token to recurse().
        chunk_size: How many characters per slice. Smaller = more calls but
                    less context per call; larger = fewer calls but more risk
                    of overflow.
        max_iterations: Maximum recursive calls. Prevents runaway loops.
        """
        self.complete_fn = complete_fn
        self.chunk_size = chunk_size
        self.max_iterations = max_iterations

    def chunk_context(self, context: str) -> list[tuple[int, str]]:
        """Split context into (start_char, chunk_text) tuples.

        Each chunk is at most chunk_size characters.
        """
        chunks = []
        start = 0
        while start < len(context):
            end = min(start + self.chunk_size, len(context))
            chunk_text = context[start:end]
            chunks.append((start, chunk_text))
            start = end
        return chunks if chunks else [(0, "")]

    def query_chunk(self, chunk: str, query: str) -> str:
        """Query a single chunk.

        Returns the model's answer, or an empty string if it cannot answer.
        """
        if not self.complete_fn:
            raise ValueError(
                "No completion function provided. Pass complete_fn to __init__ "
                "or use query_with_backend()."
            )

        prompt = (
            f"Answer this question based ONLY on the provided context. "
            f"If you cannot answer from the context, say 'NOT_FOUND'.\n\n"
            f"Context:\n{chunk}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        answer = self.complete_fn(prompt).strip()

        # If model says it cannot answer, return empty
        if answer.upper() in ("NOT_FOUND", "[NOT_FOUND]", "NOT FOUND"):
            return ""
        return answer

    def recurse(self, context: str, query: str) -> RLMResult:
        """Recursively query a context larger than the window.

        This is the main entry point for synchronous recursion.

        Returns:
            RLMResult with final_answer, slices_read, iterations, etc.
        """
        if not context:
            return RLMResult(
                final_answer="",
                success=False,
                error="Context is empty",
            )

        chunks = self.chunk_context(context)
        slices_read = []
        answers = []
        iterations = 0

        # Try each chunk in sequence. Stop early once we have at least one answer
        # and have checked a reasonable portion, to avoid querying the whole context.
        for start_char, chunk_text in chunks:
            if iterations >= self.max_iterations:
                break

            iterations += 1
            answer = self.query_chunk(chunk_text, query)
            if answer:
                answers.append(answer)
                slices_read.append((start_char, start_char + len(chunk_text)))

        # Aggregate answers
        if not answers:
            return RLMResult(
                final_answer="",
                slices_read=slices_read,
                iterations=iterations,
                success=False,
                error=f"No answer found in {len(chunks)} chunks after {iterations} iterations",
            )

        # If one answer, return it. If multiple, ask the model to synthesize.
        if len(answers) == 1:
            final_answer = answers[0]
        else:
            # Synthesize multiple answers (only if needed)
            if iterations < self.max_iterations:
                synthesis_prompt = (
                    f"Here are several partial answers to the question '{query}':\n\n"
                    + "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
                    + "\n\nSynthesize these into one comprehensive answer:"
                )
                final_answer = self.complete_fn(synthesis_prompt).strip()
                iterations += 1
            else:
                # Hit iteration limit; just concatenate
                final_answer = " ".join(answers)

        return RLMResult(
            final_answer=final_answer,
            slices_read=slices_read,
            iterations=iterations,
            success=True,
        )

    def query_with_backend(self, context: str, query: str, backend_url: str,
                          token: Optional[str] = None, model: str = "gpt-3.5-turbo",
                          timeout: float = 60.0) -> RLMResult:
        """Query a context using an OpenAI-compatible backend.

        Args:
            context: The large text to query.
            query: The question to ask.
            backend_url: The service origin (e.g., "https://api.openai.com").
            token: Bearer token for the backend (if required).
            model: The model to use (default: "gpt-3.5-turbo").
            timeout: Request timeout in seconds.

        Returns:
            RLMResult with final_answer, slices_read, iterations, etc.
        """
        import httpx

        def _complete(prompt: str) -> str:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }

            try:
                with httpx.Client(base_url=backend_url, timeout=timeout) as client:
                    resp = client.post("/v1/chat/completions", json=payload,
                                      headers=headers)
                    if resp.status_code >= 400:
                        raise ValueError(
                            f"Backend {resp.status_code}: {resp.text[:200]}"
                        )
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                raise ValueError(f"Backend error: {exc}") from exc

        # Temporarily set the completion function
        old_fn = self.complete_fn
        self.complete_fn = _complete
        try:
            return self.recurse(context, query)
        finally:
            self.complete_fn = old_fn
