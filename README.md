# awrecurse

**Answer a question over a context far larger than the model's window — recursively, with the trace of which slices were actually read.**

```bash
pip install awrecurse
```

```python
from awrecurse import RecurseClient

c = RecurseClient("https://recurse.example.com", token="...")
result = c.recurse(huge_document, "what changed in version 2.1?")
print(result["final_answer"])
print(result["slices_read"])  # which parts the model actually examined
```

```bash
awrecurse ask --file BIG.txt --query "what changed in version 2.1?"
awrecurse ask --file BIG.txt --query "..." --max-iterations 20
awrecurse ask --file BIG.txt --query "..." --json
awrecurse health          # service status
awrecurse --self-test     # prove the contract, offline
```

---

## What this is, and what it is not

It is a **client**. Recursion, chunking, aggregation and model queries stay in
the service; this is the wire contract, packaged so anything can speak it.

That split is deliberate. The alternative was lifting a 2,500-line cognitive
system with dozens of private imports into a package, which produces something
that `ModuleNotFoundError`s on your machine while reading as authoritative. A
broken package is worse than no package.

| route | what it does |
|---|---|
| `POST /recurse` | recursively answer a question about a large context |
| `GET /health` | service health and status |
| `GET /config` | service configuration and capabilities |

---

## The bug this package exists to prevent

Your model has a window. A context larger than the window doesn't raise an error.
The middle gets dropped. The model answers fluently from the beginning and end,
and the reply looks exactly like an answer drawn from the whole thing.

**The failure is a SILENCE.**

```python
# Naive approach — the model's output looks right, but:
model.complete("Here is a 200-page document:\n" + huge_doc + "\n\nQuestion: " + q)
# ↑ The middle 180 pages were dropped. The answer about the middle is wrong.
# ↑ No error, no warning, no shorter answer. It looks right.
```

The only protection is **tracking which parts of the context were actually
examined**. So every result includes `slices_read` — the list of `(start_char,
end_char)` tuples showing which chunks the model looked at. Without it, you're
blind to the exact failure case this package is written to prevent.

```python
result = c.recurse(huge_doc, "what changed in the middle section?")
print(result["slices_read"])  # [(start, end), (start, end), ...]
# If the middle section is NOT here, the answer is garbage.
```

---

## How it works: Recursive Language Model (RLM)

The algorithm (from Zhang, Kraska & Khattab, arXiv:2512.24601):

1. Split the context into slices smaller than the window.
2. Query the first slice for an answer. If found, return it and the slice index.
3. If not found, recursively query the remaining slices by aggregating their answers.
4. Cap iterations to prevent runaway loops; return the best answer found.

The result is **exact** — which parts were examined — because the model never
sees the parts that were not asked. If a question requires the middle, the
middle will be in `slices_read`.

---

## Config

The service origin comes from `--url` or `AWRECURSE_URL`; the token from
`--token` or `AWRECURSE_TOKEN`. Neither is guessed — a client that quietly
falls back to some default endpoint sends your queries somewhere you did not
choose.

```bash
export AWRECURSE_URL="https://recurse.example.com"
export AWRECURSE_TOKEN="sk-..."
awrecurse ask --file doc.txt --query "..."
```

For local recursion (no service), pass a `complete_fn` callable to
`RecurseClient`:

```python
def my_complete(prompt: str) -> str:
    # Your completion logic here
    return model.complete(prompt)

c = RecurseClient(complete_fn=my_complete)
result = c.recurse(huge_doc, "question")
```

---

## Two things it refuses to do

**Return empty on failure.** A recursion that failed raises `RecurseError`. An
empty result might mean "no answer was found" (a valid result) or "the service
is down" (a failure), and a client that returns the same for both makes a dead
backend look exactly like an unpopular query — nobody investigates unpopular
queries.

**Send an empty `Authorization` header.** No token means no header at all. An
empty Bearer is rejected differently from an absent one, and the difference
sends you debugging the auth server instead of your config.

---

## `--self-test`

Every install can prove the client still holds its contract, with no service and
no network:

```console
$ awrecurse --self-test
  PASS  recurse body is exactly the declared field set, with expected defaults
  PASS  empty inputs and unreasonable limits are refused with a reason
  PASS  chunking splits context, results track slices_read, local mode works
SELF-TEST: awrecurse ok
```

The assertions here are the load-bearing ones: the result MUST include
`slices_read`. Without it, silent truncation is invisible.

---

## Limits and trade-offs

**More calls, not cheaper.** Recursion makes multiple model calls where one
would have sufficed. If cost-per-call matters, RLM is not free.

**Not a vector database.** Recursion works well when the answer is **localized**
— a span that fits in one or two chunks. For questions that need to aggregate
across the whole document, RLM makes many calls and still synthesizes; vector
search would be better if the corpus is in a database.

**Sequential, not parallel.** The current implementation chunks and queries
sequentially. Parallelizing chunks (asking several at once) is possible but
requires different plumbing — the service would handle it, not the client.

---

## See also

- [RLM paper](https://arxiv.org/abs/2512.24601) — Zhang, Kraska & Khattab
- [awdk](https://github.com/Aitherium/awdk) — build AI agent fleets
- [awm](https://github.com/Aitherium/awm) — scoped agent memory
- [awskills](https://github.com/Aitherium/awskills) — portable agent procedures
