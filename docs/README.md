# Gwenedeg documentation

The front page is the [README](../README.md): what Gwenedeg is, what is different about it, the headline numbers, a quick
start. This index lists everything else, by who it is for. The kind says how to read a file: a **reference** describes
the code as it is; a **design** explains why and records what was measured while deciding; **history** is a dated record
that is not rewritten when the code moves on.

## Using runq

| document | kind | contents |
|---|---|---|
| [usage.md](usage.md) | reference | command line options, the load-time memory check, chat behaviour, chat templates and thinking, speculative decoding (draft model, MTP drafter, prompt lookup), KV cache types, platform and format limitations |
| [models.md](models.md) | reference | the download sets and files, Qwen3.5's recurrent state, qwen3.c's `.bin`, what is not supported |
| [../BENCHMARKS.md](../BENCHMARKS.md) | measurements | every speed and quality table with its date, setup and caveats; the comparison with llama.cpp and ik_llama.cpp; Apple M1 Pro |

## Running the server

| document | kind | contents |
|---|---|---|
| [server.md](server.md) | reference | `runq serve`: flags, endpoints and request fields, tool calls, prefix reuse, constrained decoding, speculation, agent sessions with pi, the Qwen3.8-27B setup, LiteLLM judge, limitations |
| [memory-admission.md](memory-admission.md) | design + reference | `--admit worst / bound / chunk`: leases, the live budget, parking, swapping to disk, yielding; options, tests; the measurements still open |
| [kv-disk.md](kv-disk.md) | reference | the KV cache's disk tier (`--kv-disk`): what is saved and when, file layout and locking, the MTP view's files, measured save and load times |

## Internals

| document | kind | contents |
|---|---|---|
| [library.md](library.md) | reference | the C API (`src/runq.h`): models, sequences, contexts, the forward passes; the source file map |
| [iquants.md](iquants.md) | design + reference | Q2_K and the i-quants: formats, arithmetic, AVX-512 and NEON kernels, `--repack`, measurements, what is not done |
| [kv-quant.md](kv-quant.md) | design + reference | the 8-bit and 4-bit KV caches: definition, quality (KL vs HF), speed |
| [qwen35.md](qwen35.md) | design | Qwen3.5's architecture and GGUF layout, design decisions, performance, long context |
| [DEVELOPMENT.md](DEVELOPMENT.md) | design + history | changes from qwen3.c, chat / KV / threads details, per-step benchmark notes, platform and test status, testing and benchmark scripts, CI with Dagger, rejected experiments, the step table |
| [../tests/README.md](../tests/README.md) | reference | the regression suite: groups, hooks, thresholds, goldens |
| [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) | legal | third-party code, data and formats, and their licenses |

## Open work and notes

| document | kind | contents |
|---|---|---|
| [../TODO.md](../TODO.md) | plan | open steps only; finished work is in git |
| [../IDEAS.md](../IDEAS.md) | plan | directions not scheduled |
| [notes/qwen38-speculative.md](notes/qwen38-speculative.md) | history | the lab notebook of the speculative-decoding and Qwen3.8-27B work, 2026-10-27 to 2026-11-10 |

## Conventions

- Measurements say when, on what and how they were taken. The development machine is a shared 16-vCPU AMD EPYC Genoa
  VM, so numbers carry ±15-30% run-to-run noise unless a table says otherwise; do not compare numbers across tables
  that name different sessions.
- Claims of speed against another engine exist for one case only: Qwen3.8-27B IQ3_S against mainline llama.cpp and
  ik_llama.cpp on one VM
  ([BENCHMARKS.md](../BENCHMARKS.md#qwen38-27b-iq3_s-runq-vs-llamacpp-vs-ik_llamacpp-server-workloads-2026-11-22)).
- "Step N", "S9c", "G7" and similar labels are the development steps of [TODO.md](../TODO.md) and the History table of
  [DEVELOPMENT.md](DEVELOPMENT.md#history); the old development log is in git: `git show b3e48a3:TODO.md`.
