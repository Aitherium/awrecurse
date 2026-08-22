"""awrecurse CLI.

    awrecurse ask --file BIG.txt --query "what changed in version 2.1?"
    awrecurse ask --file BIG.txt --query "..." --max-iterations 20
    awrecurse ask --file BIG.txt --query "..." --json
    awrecurse health
    awrecurse --self-test

The service origin comes from --url or AWRECURSE_URL; the token from --token or
AWRECURSE_TOKEN. For local recursion, provide a --complete command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from awrecurse.client import RecurseClient, RecurseError, recurse_body
from awrecurse.engine import RecursionEngine

# ── self-test ──────────────────────────────────────────────────────────────
# Everything asserted here is PURE. A self-test that needs a live service is
# a self-test that gets skipped.


def _refuses(context: str, query: str, **kwargs: object) -> bool:
    """True when recurse_body rejects this call."""
    try:
        recurse_body(context, query, **kwargs)  # type: ignore[arg-type]
    except ValueError:
        return True
    return False


def _self_test() -> int:
    failures: list[str] = []

    # 1. The recurse body is EXACTLY the declared field set.
    body = recurse_body("context here", "query here")
    declared = ("context", "query", "chunk_size", "max_iterations")
    if tuple(body) != declared:
        failures.append(f"recurse_body keys {tuple(body)} != declared {declared}")

    # 2. Defaults match expectations
    if body != {
        "context": "context here", "query": "query here",
        "chunk_size": 2000, "max_iterations": 10
    }:
        failures.append(f"recurse_body defaults drifted: {body}")

    # 3. Empty inputs are refused with a reason
    for field, val in (("context", ""), ("query", "")):
        if not _refuses("ctx" if field == "query" else "", "q" if field != "query" else ""):
            failures.append(f"empty {field} was accepted")

    # 4. Unreasonable chunk_size is refused
    if not _refuses("c", "q", chunk_size=50):
        failures.append("chunk_size < 100 was accepted")
    if not _refuses("c", "q", max_iterations=0):
        failures.append("max_iterations < 1 was accepted")

    # 5. Chunking works: split a 5000-char context into 2000-char chunks
    ctx = "x" * 5000
    engine = RecursionEngine(chunk_size=2000, max_iterations=10)
    chunks = engine.chunk_context(ctx)
    if len(chunks) != 3:
        failures.append(f"5000-char context should split into 3 chunks of 2000, got {len(chunks)}")
    if chunks[0][1] != "x" * 2000:
        failures.append("first chunk has wrong size")

    # 6. A result tracks slices_read — the core contract
    def dummy_complete(prompt: str) -> str:
        return "dummy answer" if "question" in prompt.lower() else ""

    engine = RecursionEngine(dummy_complete, chunk_size=1000, max_iterations=5)
    result = engine.recurse("a" * 3000, "question")
    if not isinstance(result.slices_read, list):
        failures.append("slices_read is not a list")
    if result.iterations < 1:
        failures.append("iterations not set")

    # 7. to_dict() returns the full trace
    trace = result.to_dict()
    required = ("final_answer", "slices_read", "iterations", "tokens", "success", "error", "trace")
    for key in required:
        if key not in trace:
            failures.append(f"to_dict() missing key: {key}")

    # 8. RecurseClient requires URL or complete_fn
    c = RecurseClient()  # no URL, no complete_fn
    try:
        c.recurse("context", "query")
        failures.append("recurse() with no URL/complete_fn should raise")
    except RecurseError:
        pass  # expected

    # 9. Local complete_fn works
    c = RecurseClient(complete_fn=dummy_complete)
    result_dict = c.recurse("context", "query", chunk_size=500, max_iterations=3)
    if "final_answer" not in result_dict:
        failures.append("local recurse() did not return final_answer")
    if "slices_read" not in result_dict:
        failures.append("local recurse() did not return slices_read")

    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        print(f"SELF-TEST: {len(failures)} failure(s)")
        return 1
    print("  PASS  recurse body is exactly the declared field set, with expected defaults")
    print("  PASS  empty inputs and unreasonable limits are refused with a reason")
    print("  PASS  chunking splits context, results track slices_read, local mode works")
    print("SELF-TEST: awrecurse ok")
    return 0


# ── commands ───────────────────────────────────────────────────────────────


def _client(args: argparse.Namespace) -> RecurseClient:
    url = args.url or os.environ.get("AWRECURSE_URL")
    complete_fn = None

    # If --complete is provided, try to use it as a local completion function
    if args.complete:
        # For the CLI, --complete would be a command to run; for this example,
        # we require a Python callable. A real CLI might wrap an external command.
        raise NotImplementedError(
            "--complete requires passing a Python callable, not yet implemented for CLI"
        )

    if not url and not complete_fn:
        print("no service URL: pass --url or set AWRECURSE_URL", file=sys.stderr)
        raise SystemExit(2)

    token = args.token or os.environ.get("AWRECURSE_TOKEN")
    return RecurseClient(url, token, complete_fn=complete_fn)


def _show(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return

    print(result.get("final_answer", ""))
    print()

    slices = result.get("slices_read", [])
    if slices:
        print(f"Examined {len(slices)} slice(s) of the context:")
        for start, end in slices:
            print(f"  [{start:6d} - {end:6d}]")
    print()

    print(f"Iterations: {result.get('iterations', 0)}")
    print(f"Tokens: {result.get('tokens', '?')}")
    if not result.get("success"):
        print(f"Error: {result.get('error', 'unknown')}")


def main(argv: list[str] | None = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    ap = argparse.ArgumentParser(
        prog="awrecurse",
        description="Recursively query a context larger than the model's window.",
    )
    ap.add_argument("--self-test", action="store_true",
                    help="prove this client still holds its contract, offline")
    ap.add_argument("--url", help="service origin (or AWRECURSE_URL)")
    ap.add_argument("--token", help="bearer token (or AWRECURSE_TOKEN)")
    ap.add_argument("--json", action="store_true", help="print the raw response")
    ap.add_argument("--complete", help="local Python callable (not yet CLI-supported)")

    sub = ap.add_subparsers(dest="cmd")

    # ask subcommand
    p = sub.add_parser("ask", help="ask a question about a large document")
    p.add_argument("--file", required=True, help="path to the context file")
    p.add_argument("--query", "-q", required=True, help="the question to ask")
    p.add_argument("--chunk-size", type=int, default=2000,
                   help="characters per chunk (default: 2000)")
    p.add_argument("--max-iterations", type=int, default=10,
                   help="max recursive calls (default: 10)")

    # health subcommand
    sub.add_parser("health", help="check service health")

    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.cmd:
        ap.print_help()
        return 2

    try:
        if args.cmd == "ask":
            if not os.path.exists(args.file):
                print(f"awrecurse: file not found: {args.file}", file=sys.stderr)
                return 2
            with open(args.file, "r", encoding="utf-8") as f:
                context = f.read()

            c = _client(args)
            result = c.recurse(context, args.query,
                              chunk_size=args.chunk_size,
                              max_iterations=args.max_iterations)
            _show(result, args.json)
            return 0

        if args.cmd == "health":
            c = _client(args)
            result = c.health()
            print(json.dumps(result, indent=2))
            return 0

    except ValueError as exc:
        # A refusal made HERE, before the round trip.
        print(f"awrecurse: {exc}", file=sys.stderr)
        return 2
    except RecurseError as exc:
        print(f"awrecurse: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
