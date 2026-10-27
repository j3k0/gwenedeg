# `runq serve`: the OpenAI-compatible server

The reference for [Gwenedeg](../README.md)'s server: flags, endpoints, behaviour, the LiteLLM
router setup and measurements (on the 16-vCPU AMD EPYC Genoa VM of [BENCHMARKS.md](../BENCHMARKS.md),
shared, so the numbers are noisy). The design is described in the header comments of `src/engine.c` (scheduler,
slot pool) and `src/constrain.c` (constrained decoding); the full measurements are in Steps 9 and 10b of the
development log ([DEVELOPMENT.md, History](DEVELOPMENT.md#history)), the tuning of the defaults in
[DEVELOPMENT.md](DEVELOPMENT.md#design-notes-and-rejected-experiments) (Server).

An OpenAI-compatible HTTP server in the same binary (POSIX sockets; on Windows it reports that it is not
available). One model per process, bound to 127.0.0.1 by default, no auth unless `--api-key`:

```sh
./runq serve models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf             # http://127.0.0.1:8080/v1
curl -s localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Hi!"}], "max_tokens": 32}'
```

| flag | default | |
|---|---|---|
| `--host`, `--port` | 127.0.0.1, 8080 | `--port 0`: any free port (printed: `listening on http://...`) |
| `--api-key KEY` | none | require `Authorization: Bearer KEY` (not on `/health`) |
| `--name ID` | the file name | the model id in `/v1/models` and responses (the request's `model` is accepted whatever it is) |
| `-c`, `--ctx N` | 8192 | positions per sequence (the context limit of a request) |
| `--parallel N` | 16 | requests running at once (their rows batched in each forward pass); more wait in the queue |
| `--slots N` | parallel + 4 | sequences in the pool: one per running request, the idle ones keep their tokens as prefix caches |
| `--kv-mem GiB` | half the RAM | bound for slots x ctx x KV bytes per position (115 KB f16): `--slots` (and `--parallel`) are lowered to fit; the memory is touched only as used |
| `-b N`, `--chunk N` | 512, 64 | prompt tokens per forward pass when no request is decoding / while others decode |
| `--prefix-copy N` | 32 | copy a running request's cached prompt prefix into a new request's slot if that saves N tokens (0: never) |
| `-k` | f16 | KV cache type (as for the CLI) |
| `--queue N` | 64 | requests waiting beyond the free running places at most (more: 503, `Retry-After: 1`) |
| `--max-conns N`, `--max-body B` | 128, 4 MB | connections (more: 503), request body size (more: 413) |
| `--timeout S` | 600 | per request incl. queueing (504; 0 = none) |
| `-t`, `-p`, `-n` | 1.0, 1.0, 0 | defaults for requests without temperature / top_p / max_tokens (0: up to the context) |
| `--top-k N` | 0 (off) | default for requests without top_k. `-t` / `-p` / `--top-k` not given: the model's recommended values if its GGUF has them (`general.sampling.temp` / `top_p` / `top_k`, e.g. Qwen3.8-27B 1.0 / 0.95 / 20) |
| `--threads N` | `OMP_NUM_THREADS`, else nproc - nproc/8 | threads per forward pass |
| `-v`, `--verbose` | off | log request bodies and responses |

Endpoints: `POST /v1/chat/completions` (also `/chat/completions`), `GET /v1/models`, `GET /v1/models/{id}`,
`GET /health` (also `/v1/health`: status, running and queued requests). Chat completions take `messages`
(system / developer / user / assistant; content a string or an array of `{"type": "text"}` parts),
`max_tokens` / `max_completion_tokens`, `temperature` (0 = greedy), `top_p`, `top_k` (an extension, as vLLM's
and llama.cpp's: an integer, 0 or -1 = off; top-p then applies among the k), `seed`, `stop` (a string or
up to 16), `n` (1 only), `stream` (SSE `data:` chunks, `data: [DONE]`, a final usage chunk with
`stream_options.include_usage`), `logprobs` / `top_logprobs` (up to 20; not with `stream`), `response_format`
(`text`; `json_schema`, below; `json_object` is rejected with a message), thinking with `reasoning_effort`
low / medium / high or `chat_template_kwargs.enable_thinking` (off by default; the reasoning comes back as
`reasoning_content`), and `cache_prompt: false` (llama.cpp's flag: prefill from scratch). `usage` counts
prompt, completion and cached prompt tokens (`prompt_tokens_details.cached_tokens`); `finish_reason` is
`stop` or `length`. Errors are OpenAI error objects: 400 (invalid request; a prompt longer than the context
has `code: context_length_exceeded`), 401, 404, 405, 411 (chunked uploads), 413, 503 (queue full / shutting
down), 504. Not supported: tools / function calls, images, `n > 1`, `json_object`, penalties (ignored).

- **Conversation**: rendered with Qwen3's template (the model's variant: README, Chat templates) in the CLI chat's form (`runq -m chat`), so a greedy
  request gives the same text as the CLI for the same messages (tested bitwise, also with a seed: the RNG
  draws as the CLI's chat does for a first turn). Earlier assistant messages lose their `<think>` block.
- **Prefix reuse**: a new request takes the idle slot with the longest common token prefix, truncated
  to it, and prefills only the rest (if that saves at least 32 tokens or half the prompt; else the least
  recently used idle slot). If a running request's slot shares a longer prefix, it is copied
  (`rq_seq_copy`, ~0.6-1.6 ms for LiteLLM's 253-token rubric vs ~110 ms to prefill it); if a request still
  being prefilled will share it, the new one waits for that part and copies it then (a burst of
  classifications prefills the rubric once). Logits are bit-identical to a fresh prefill (tested). A second
  turn re-prefills only the new message; LiteLLM's classifier rubric stays cached.
- **Constrained decoding** (`response_format: json_schema`), for schemas with a finite set of outputs:
  `enum`, `const`, `boolean`, `null`, `integer` with `minimum` / `maximum` (up to 256 values), objects of
  those (nested; `required`, optional properties; property order as in the schema; no additional
  properties), `anyOf` / `oneOf`, `$ref` into `$defs` / `definitions`; up to 4096 outputs. Each output is
  written as Python's `json.dumps` writes it (`{"tier": "SIMPLE"}`: what Qwen3 itself writes for a small
  object). Where all valid continuations agree, their canonical tokens are forced without a forward pass
  of their own (`{"tier": "` goes into the prompt's prefill, `PLE"}` after the decision is never fed);
  decisions are masked at the byte level (every token whose bytes keep the output a prefix of a valid
  one), so a LiteLLM classification is one prefill: ~20 ms with the rubric cached. Other schemas get a
  400 naming the unsupported part.
- **Concurrency** (continuous batching, Step 10b): one engine thread runs every request; each step is one
  batched forward pass over the next token of every decoding request plus prompt chunks of requests being
  prefilled (up to `--chunk` 64 tokens while others decode, so streams keep flowing: gaps between tokens
  p99 ~60-70 ms, at most ~90 ms, while a 2,000-token prompt is prefilled, where they used to stop for 7-12 s; `-b` 512 when nothing decodes; the budget is
  shared, so a short prompt does not wait for a long one), then the sampling of every row with each
  request's settings and token mask. Requests join at the next step (FIFO, at most `--parallel`) and leave
  when done; a request's output does not depend on what it is batched with (each has its own sequence and
  RNG; tested against `runq -m chat` and against the same request alone). Connections have their own
  threads, so `/health` and queueing never wait on a forward pass. A client that disconnects cancels its
  request at the next step (the others continue). SIGINT / SIGTERM: stop accepting, reject the queue
  (503), let the running requests finish (up to 10 s), exit.
- **Throughput** (`scripts/bench_serve.py`, 12 threads, streamed 128-token replies, before = Step 9's
  one-at-a-time engine): Qwen3-0.6B Q4_K_M 1 / 4 / 8 / 16 clients 309 / 740 / 914 / 1,185 tok/s in
  aggregate (before ~265-300 whatever the count), time to first token at 16 clients ~0.1 s (before 3.5 s);
  Q8_0 184 / 577 / 911 / 1,223 (before ~200). LiteLLM-judge-like load (rubric cached, 16 clients): 0.6B
  ~62-69 classifications/s (before ~46-51), 1.7B Q8_0 ~30-34 (before ~24): a classification is ~30-60
  prompt tokens and no decode step, so prefill-bound; batching helps by making the passes larger.
- **Logging**: one line per request on stderr: time, client, method, path, status, prompt / cached /
  completion tokens, queue wait, prefill and total time (`--verbose`: the bodies too).

As the judge of LiteLLM's complexity router (`auto_router/complexity_router`, `classifier_type: llm`,
LiteLLM 1.105): LiteLLM sends the rubric as the system message, `\nClassify this message:\n<prompt>` as the
user message, and `response_format` = its `TierClassification` model's JSON schema (an inline enum of the
four tiers); runq answers `{"tier": "..."}`. The judge needs no thinking and `temperature: 0`:

```yaml
model_list:
  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        classifier_type: llm
        classifier_context_window_size: 0     # judge the current message only (shorter prompts)
        classifier_llm_config:
          model: runq-judge
          timeout_ms: 10000                    # default 3000; a timeout opens a 30 s circuit breaker
        tiers: { SIMPLE: small-model, MEDIUM: small-model, COMPLEX: big-model, REASONING: big-model }
  - model_name: runq-judge
    litellm_params:
      model: openai/runq                       # any name
      api_base: http://127.0.0.1:8080/v1
      api_key: none
      temperature: 0
```

`LITELLM_VENV=/path/to/venv scripts/litellm_e2e.sh` runs this chain with a scratch LiteLLM proxy from a venv
with LiteLLM's proxy installed (tested with 1.105.0; runq as the judge and as the tier models; a second pass sends the same prompts concurrently). Judge latency on the 16-vCPU VM (8 threads,
Q8_0 GGUF): ~180 ms for the first request (the 265-token prompt), then 15-27 ms (the rubric cached). Quality (`scripts/eval_router.py`, 80 labeled
prompts, `scripts/router_eval.jsonl`, written for this project; Step 9): Qwen3-0.6B is a weak judge, 51-71% depending on rubric and quantization (it
almost never says REASONING); Qwen3-1.7B Q8_0 79-88% at ~55 ms per classification; Qwen3-4B Q4_K_M
79-92% at ~150 ms. Later, with the bigger models (Step 11): 1.7B Q8_0 88% at ~50 ms, Qwen3-4B-Instruct-2507
Q4_K_M 88-92% at ~130 ms, Qwen3-8B Q4_K_M 92% at ~260 ms.

## Limitations

- Output is constrained only to JSON schemas with a finite set of outputs (no `json_object`, no free strings);
  no tools / function calls, images, `n > 1`; penalties are ignored.
- No TLS: bind it to localhost or a trusted network (`--api-key` is a shared secret over plain HTTP).
- The KV memory is a fixed pool (`--slots` x `-c`, no paging or preemption): a request keeps its slot until it ends.
- One model per process. POSIX only (on Windows the server reports that it is not available).

