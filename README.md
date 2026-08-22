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

<!-- aither-ecosystem:start GENERATED from the ecosystem registry. Edits here are overwritten; change the registry instead. -->

## The aw family

Standalone tools that share one idea: **replace something you would otherwise have to _trust_ with something you can _check_.**

Each installs on its own, works offline, and needs no account.

| | instead of trusting | you check |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | a framework's idea of how your agents should run | one loop you can read, pointed at a backend you already pay for |
| [awskills](https://github.com/Aitherium/awskills) | that an agent knows your procedure | the procedure written down, versioned, and loadable by any agent |
| [awm](https://github.com/Aitherium/awm) | that memory stayed in its lane | tenant:user:project scopes, so a write cannot cross a boundary |
| [awnode](https://github.com/Aitherium/awnode) | a vendor's cloud with every prompt | a local gateway routing to backends you chose |
| [awgraph](https://github.com/Aitherium/awgraph) | that grep found everything | an AST + tree-sitter call graph an agent can traverse |
| [awgit](https://github.com/Aitherium/awgit) | that no one else is editing this file | a lease, refused at commit time if you do not hold it |
| [awseal](https://github.com/Aitherium/awseal) | that the artifact came from who you think | an Ed25519 seal — the key that verifies is not the key that forges |
| [awshare](https://github.com/Aitherium/awshare) | that the download is intact | content-addressed bundles, verified on fetch |
| [awnest](https://github.com/Aitherium/awnest) | that there is a person on the other end | a verdict with evidence, where "we could not tell" is not "yes" |
| [awnboard](https://github.com/Aitherium/awnboard) | a share link anyone who sees it can use | an invitation addressed to one person, for one gate, revocable |
| [awnix](https://github.com/Aitherium/awnix) | that the box is what you left it as | an immutable image you built, with atomic rollback |
| [awrecover](https://github.com/Aitherium/awrecover) | that the restore worked | a restore that fully lands or does not land at all |
| [awrelay](https://github.com/Aitherium/awrelay) | a SaaS in the middle of your agents | findings, alerts and coordination over your own transport |
| [awmail](https://github.com/Aitherium/awmail) | a mailbox somebody else can read | mail your agents send and receive over your own server |
| [awfind](https://github.com/Aitherium/awfind) | one vendor's idea of the web | results from whichever providers you configured |
| [awbrowse](https://github.com/Aitherium/awbrowse) | that the page said what you were told | the render, the DOM and the requests it made |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | a vendor's quantisation defaults | sub-byte KV cache kernels you can benchmark yourself |
| [AitherZero](https://github.com/Aitherium/AitherZero) | a pile of scripts nobody has numbered | numbered, discoverable automation with declarative playbooks |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | what a page tells your browser to do | a federated search and desktop bridge you host |
| [awreason](https://github.com/Aitherium/awreason) | a confident paragraph | the phases it went through, and every tool call it made to get there |
| **awrecurse** _(you are here)_ | that everything you pasted in was actually read | which slices it opened, and what it concluded from each |
| [awprism](https://github.com/Aitherium/awprism) | the first explanation that fits | the ranked alternatives, and the observation that separates them |
| [awrepl](https://github.com/Aitherium/awrepl) | what the agent believes the value is | the value, printed from the live session |
| [awresearch](https://github.com/Aitherium/awresearch) | a summary of pages nobody opened | every claim against the source it came from |
| [awkno](https://github.com/Aitherium/awkno) | that the docs site is up, or that you remember the family | the whole ecosystem in your terminal, with no network at all |

[**awnix**](https://github.com/Aitherium/awnix) is the ground floor — A Linux you can hand to an agent — immutable base, capabilities included.

## The Aitherium ecosystem

Every repository here is public. Each publishes an `aither-manifest.json` beside its page, so any surface can read every sibling's — the network is browsable from any node in it.

| repo | what it is | pages |
|---|---|---|
| [awdk](https://github.com/Aitherium/awdk) | Build AI agent fleets — 3 lines, any backend, local or cloud | [docs](https://aitherium.github.io/awdk/) |
| [awskills](https://github.com/Aitherium/awskills) | Portable agent skills — self-contained procedures an agent loads on demand | [docs](https://aitherium.github.io/awskills/) |
| [awm](https://github.com/Aitherium/awm) | A portable, scoped agent memory | [docs](https://aitherium.github.io/awm/) |
| [awnode](https://github.com/Aitherium/awnode) | A lightweight local gateway — bridges your apps to the AI backends you chose | [docs](https://aitherium.github.io/awnode/) |
| [awrun](https://github.com/Aitherium/awrun) | A priority-aware queue and dispatcher for agentic runs and ad-hoc CI builds | [docs](https://aitherium.github.io/awrun/) |
| [awgraph](https://github.com/Aitherium/awgraph) | A semantic code graph for agents — AST + tree-sitter, call graphs | [docs](https://aitherium.github.io/awgraph/) |
| [awgit](https://github.com/Aitherium/awgit) | Semantic version control on top of git — edit-ops and leases | [docs](https://aitherium.github.io/awgit/) |
| [awseal](https://github.com/Aitherium/awseal) | Sign an artifact so a stranger can verify it | [docs](https://aitherium.github.io/awseal/) |
| [awshare](https://github.com/Aitherium/awshare) | Publish an artifact and fetch it back verified | [docs](https://aitherium.github.io/awshare/) |
| [awnest](https://github.com/Aitherium/awnest) | Prove there is a human before you let them into the nest | [docs](https://aitherium.github.io/awnest/) |
| [awnboard](https://github.com/Aitherium/awnboard) | A front gate you can put in front of anything, and hand someone the key to | [docs](https://aitherium.github.io/awnboard/) |
| [awnix](https://github.com/Aitherium/awnix) | A Linux you can hand to an agent — immutable base, capabilities included | [docs](https://aitherium.github.io/awnix/) |
| [awrecover](https://github.com/Aitherium/awrecover) | Labelled snapshots with an all-or-nothing restore | [docs](https://aitherium.github.io/awrecover/) |
| [awrelay](https://github.com/Aitherium/awrelay) | Portable agent messaging — findings, alerts, coordination | [docs](https://aitherium.github.io/awrelay/) |
| [awmail](https://github.com/Aitherium/awmail) | Give an agent an email address — send, and actually receive | [docs](https://aitherium.github.io/awmail/) |
| [awfind](https://github.com/Aitherium/awfind) | A portable search client — query, results, ranking | [docs](https://aitherium.github.io/awfind/) |
| [awbrowse](https://github.com/Aitherium/awbrowse) | A portable browser client — navigate, console, network, DOM, screenshot | [docs](https://aitherium.github.io/awbrowse/) |
| [aitherkvcache](https://github.com/Aitherium/aitherkvcache) | Near-optimal KV cache quantization for LLM inference — sub-byte compression | [docs](https://aitherium.github.io/aitherkvcache/) |
| [AitherZero](https://github.com/Aitherium/AitherZero) | PowerShell 7+ automation framework — numbered, self-describing scripts | [docs](https://aitherium.github.io/AitherZero/) |
| [AitherConnect](https://github.com/Aitherium/AitherConnect) | Browser extension — federated AI search, page context, and the Living OS overlay | [docs](https://aitherium.github.io/AitherConnect/) |
| [awreason](https://github.com/Aitherium/awreason) | A portable reasoning client — sessions, phases, thoughts, and the chain that produced the answer | [docs](https://aitherium.github.io/awreason/) |
| **awrecurse** _(you are here)_ | Answer a question over a context far larger than the window — recursively, with the trace kept | [docs](https://aitherium.github.io/awrecurse/) |
| [awprism](https://github.com/Aitherium/awprism) | Turn a failure into ranked hypotheses — and say what would confirm each one | [docs](https://aitherium.github.io/awprism/) |
| [awrepl](https://github.com/Aitherium/awrepl) | A REPL an agent can actually use — state that survives between turns | [docs](https://aitherium.github.io/awrepl/) |
| [awresearch](https://github.com/Aitherium/awresearch) | Ask a research question, get a cited report you can check | [docs](https://aitherium.github.io/awresearch/) |
| [awkno](https://github.com/Aitherium/awkno) | The man page for the Aither World — every brick, stack and law, offline | [docs](https://aitherium.github.io/awkno/) |

<div id="aither-constellation" data-self="awrecurse"></div>
<script src="aither-constellation.js"></script>

<!-- aither-ecosystem:end -->
