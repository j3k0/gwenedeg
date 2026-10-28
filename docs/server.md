# `runq serve`: the OpenAI-compatible server

The reference for [Gwenedeg](../README.md)'s server: flags, endpoints, behaviour, the LiteLLM
router setup and measurements (on the 16-vCPU AMD EPYC Genoa VM of [BENCHMARKS.md](../BENCHMARKS.md),
shared, so the numbers are noisy). The design is described in the header comments of `src/engine.c` (scheduler,
slot pool), `src/constrain.c` (constrained decoding) and `src/tools.c` (tool calls); the full measurements are in Steps 9 and 10b of the
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
| `--kv-mem GiB` | half the RAM, at most MemAvailable after the load | bound for the pool: slots x (the KV cache of ctx positions + for Qwen3.5 / 3.8 the recurrent state and its `--checkpoints` copies); `--slots` (and `--parallel`) are lowered to fit; the memory is touched only as used. The startup line shows the size of a slot (e.g. Qwen3-0.6B f16: 115 KB per position, 0.9 GiB at 8192; Qwen3.5-0.8B at 2048 positions: 12 KB per position + 19.3 MiB of state x (1 + 4) = 120.5 MiB; Qwen3.8-27B at 8192: 0.5 GiB + 150 MiB x 5 = 1.23 GiB) |
| `--checkpoints N` | 4 | Qwen3.5 / 3.8: copies of the recurrent state a slot keeps (1..32; ~19 MiB each for the 0.8B, ~150 MiB for the 27B), where an agent's next requests leave the cache (below, Prefix reuse); counted in the slot's size and in the load check |
| `--mem GiB` | MemAvailable | the load check (README, Memory): refuse to load if the weights, the runtime and one sequence of `-c` positions do not fit; `0` or `force`: no check |
| `--page-cache auto\|drop\|keep` | auto | the model file's page cache after the load (auto: dropped if the file is >= 1/4 of the RAM) |
| `-b N`, `--chunk N` | 512, 64 | prompt tokens per forward pass when no request is decoding / while others decode |
| `--prefix-copy N` | 32 | copy a cached prompt prefix into a new request's slot if that saves N tokens (a running request's; an idle slot's that would otherwise be cut, slot affinity below; 0: never) |
| `-k` | f16 | KV cache type (as for the CLI) |
| `--queue N` | 64 | requests waiting beyond the free running places at most (more: 503, `Retry-After: 1`) |
| `--max-conns N`, `--max-body B` | 128, 4 MB | connections (more: 503), request body size (more: 413) |
| `--timeout S` | 600 | per request incl. queueing (504; 0 = none) |
| `-t`, `-p`, `-n` | 1.0, 1.0, 0 | defaults for requests without temperature / top_p / max_tokens (0: up to the context) |
| `--top-k N` | 0 (off) | default for requests without top_k. `-t` / `-p` / `--top-k` not given: the model's recommended values if its GGUF has them (`general.sampling.temp` / `top_p` / `top_k`, e.g. Qwen3.8-27B 1.0 / 0.95 / 20) |
| `--threads N` | `OMP_NUM_THREADS`, else nproc - nproc/8 | threads per forward pass |
| `-D`, `--draft FILE` | none | speculative decoding with a draft model of the same vocabulary (e.g. Qwen3-0.6B Q4_K_M for Qwen3-1.7B, Qwen3.5-0.8B Q4_K_M for Qwen3.5 / 3.8; refused otherwise, or if its context is shorter than `-c`): its guesses are verified in the batched passes, the output is exactly plain decoding's (below, Speculative decoding) |
| `-G N`, `--tau T` | 3, 0 | with `--draft`: the draft model's guesses per pass at most (0..16); stop guessing at a draft top-1 probability < T (0: off) |
| `--lookup N`, `--lookup-min M` | 3 with `--draft` in auto mode, else 0; auto: by length, fixed: 2 | prompt lookup: up to N guesses copied after the latest longest match of the last tokens earlier in the context (no model, no memory; 0: off); the shortest match used (2..16) |
| `--spec auto\|fixed\|off` | auto with `--draft` / `--lookup`, else off | the default of requests: **auto** chooses the source and the guess count per pass from the cost model, the position and the batch (plain steps when no guess pays off); **fixed** speculates as the CLI's `-G` / `--lookup` (the lookup when it matches, else the model); off: plain decoding. Requests may override (`speculative`) |
| `--spec-calibrate N` | 8192 | the cost model's calibration at the start (both models, on slot 0's empty sequences: decode steps and 2-16-row verify segments at 16 and min(N, -c) positions, ~1-5 s; 0: estimates from the models' sizes); the startup line shows it |
| `-v`, `--verbose` | off | log request bodies and responses |

Endpoints: `POST /v1/chat/completions` (also `/chat/completions`), `GET /v1/models`, `GET /v1/models/{id}`,
`GET /health` (also `/v1/health`: status, running and queued requests). Chat completions take `messages`
(system / developer / user / assistant, and with Qwen3.5 / Qwen3.8 tool, with assistant `tool_calls`; content a string or an
array of `{"type": "text"}` parts), `tools` / `tool_choice` / `parallel_tool_calls` (Qwen3.5 / Qwen3.8: below),
`max_tokens` / `max_completion_tokens`, `temperature` (0 = greedy), `top_p`, `top_k` (an extension, as vLLM's
and llama.cpp's: an integer, 0 or -1 = off; top-p then applies among the k), `seed`, `stop` (a string or
up to 16), `n` (1 only), `stream` (SSE `data:` chunks, `data: [DONE]`, a final usage chunk with
`stream_options.include_usage`; a comment line `: keep-alive` after 15 s without a chunk, e.g. during a long prefill,
so clients' idle timeouts do not end the stream: pi's `httpIdleTimeoutMs` is 300 s), `logprobs` / `top_logprobs` (up to 20; not with `stream`), `response_format`
(`text`; `json_schema`, below; `json_object` is rejected with a message), thinking with `reasoning_effort`
low / medium / high / xhigh (`none` / `minimal`: off) or `chat_template_kwargs.enable_thinking` (off by default,
except Qwen3.8: below; the reasoning comes back as `reasoning_content`), `cache_prompt: false` (llama.cpp's flag: prefill from scratch)
and `force_reply` (a benchmark hook, `scripts/replay_sessions.py`: the reply is not generated but given, as the
model's raw text or, with the Qwen3.5 / 3.8 templates, as an assistant message `{content, reasoning_content,
tool_calls}` rendered as the template renders an earlier reply; its tokens are output and parsed as if sampled and
fed after the prompt in passes of several rows, then the reply ends as on the end token; not with `json_schema` or
`logprobs`), `speculative` (an extension, below: `false`, `true`, `"off"` / `"auto"` / `"fixed"` or
`{"mode", "k", "tau", "lookup", "lookup_min"}`; the fields not given are the server's). `usage` counts
prompt, completion and cached prompt tokens (`prompt_tokens_details.cached_tokens`); responses (and the stream's
usage chunk) also carry llama.cpp's `timings` (`prompt_n`: tokens prefilled, `prompt_ms`, `predicted_n` /
`predicted_ms`, `cache_n`) and runq's: `slot`, `slot_tokens` (its length when the request took it), `common_tokens`
(its common prefix with the prompt: below `slot_tokens`, the history was sent differently from what was generated;
above `cache_n`, the recurrent state rolled back to a checkpoint), `copied`, `queue_ms`, and with speculation
llama.cpp's `draft_n` / `draft_n_accepted` (guesses verified / accepted) and `speculative` (below); `finish_reason` is
`stop`, `length` or `tool_calls`. Errors are OpenAI error objects: 400 (invalid request; a prompt longer than the context
has `code: context_length_exceeded`), 401, 404, 405, 411 (chunked uploads), 413, 503 (queue full / shutting
down), 504. Not supported: tools with Qwen3's templates (400), `tool_choice` required / a named function, the legacy
`functions`, images, `n > 1`, `json_object`, penalties (ignored).

- **Conversation**: rendered with Qwen3's template (the model's variant: README, Chat templates) in the CLI chat's form (`runq -m chat`), so a greedy
  request gives the same text as the CLI for the same messages (tested bitwise, also with a seed: the RNG
  draws as the CLI's chat does for a first turn). Earlier assistant messages lose their `<think>` block.
- **Qwen3.8** (its template, detected by `reasoning_effort` in it): thinking is **on** by default with the effort
  `xhigh`; `reasoning_effort` (or `chat_template_kwargs.reasoning_effort`, which wins) `low` / `medium` / `high`
  (= `xhigh`) / `xhigh` picks the reasoning-effort instruction the system block starts with (medium: none; low: a
  short one), `none` / `minimal` or `chat_template_kwargs.enable_thinking: false` turn thinking off. The system and
  developer messages at the start are merged (`\n`); one after them is a 400 (the template raises). Earlier
  assistant messages keep their reasoning, as the template renders them by default: `"<think>\n" reasoning
  "\n</think>\n\n" content`, the reasoning from the message's `reasoning_content` (a string; then neither may
  contain `</think>`), else from its content split at its last `</think>` (none: an empty think block). Tested
  (`q35/synth-q35-27b-f32/server38/*`): each request's prompt == `runq -m render` of its messages, the greedy reply
  == the CLI's.
- **Tool calls** (`src/tools.c`; the Qwen3.5 and Qwen3.8 templates; Qwen3's older variants use another format and get a
  400). A request with `tools` (OpenAI's function tools), a `tool` message or an assistant message with `tool_calls` is
  rendered exactly as the model's template renders it (HF's `apply_chat_template`; tested on every case of
  `tests/golden/chat_render.json` with a request): the tool list in the system block (each tool as transformers'
  `tojson`, i.e. Python's `json.dumps`, then the template's instructions and the system text; Qwen3.8 merges system and
  developer messages and puts its effort instruction first), consecutive `tool` messages as one user turn of
  `<tool_response>` blocks, earlier calls as `<tool_call><function=NAME><parameter=P>value</parameter>...` (arguments
  given as a JSON string, OpenAI's form, are parsed first, as vLLM does; a string value as is, else JSON for Qwen3.8,
  Python's `str()` of a scalar for Qwen3.5: its template's). Earlier assistant messages keep their reasoning
  (`reasoning_content`, else a `<think>` block in the content) where the template does: Qwen3.8 always, Qwen3.5 after the
  last user query (in an agent loop: every reply since the user's message), so the history matches what was generated
  and a turn prefills only the tool results (pi: 1,095 of 1,130 tokens cached on the second request). A system message
  after the first other one, or (Qwen3.5) no user message at all, is a 400 (the templates raise); `developer` is taken
  as `system`; text parts are joined without a separator (the templates' `render_content`). The reply's content (after
  the reasoning: a call inside `<think>` stays reasoning) is parsed as it is generated: text before a `<tool_call>` is
  `content` (its trailing whitespace dropped), each `<function=NAME>` block becomes a `tool_calls` item (`id`
  `call_<16 hex>`, `arguments` a JSON object written as `json.dumps` writes it), each parameter (one leading and trailing
  newline removed) converted to its type in the tool's schema (`type`, a list of types or `anyOf` / `oneOf`): `null` if
  allowed and the text is `null`, a string if allowed, else an integer / number literal, `true` / `false` (any case), or
  any JSON for object / array; otherwise (an unknown parameter or tool, a value that does not convert) a string. After
  the calls the model may only add whitespace and more calls: other text ends the generation (the template asks for "NO
  suffix"); `parallel_tool_calls: false` ends it after the first call. A block that does not parse (no
  `<function=...>`, text between the parts) is content if no call came before it, and ends the generation after one; a
  block cut by the end of the reply counts if its `</function>` was generated. Non-streamed: `message.tool_calls`,
  `content` null if empty, `finish_reason: tool_calls` (`length` if the limit cut the reply); streamed: the content
  first, then each call whole in one `tool_calls` delta (`index`, `id`, `type`, `function.name`, `function.arguments`).
  `tool_choice: none` renders the tools but does not parse the reply (the XML stays in `content`; the model is not
  stopped from writing it); `auto` (the default) parses. `response_format` `json_schema` with tools (parsed) is a 400.
  Not done: constraining the parameter values by the schema while decoding.
  With pi (`openai-completions`), a `models.json` provider such as
  `{"baseUrl": "http://127.0.0.1:8080/v1", "api": "openai-completions", "apiKey": "none", "compat":
  {"supportsDeveloperRole": false, "supportsReasoningEffort": true, "supportsStore": false, "maxTokensField":
  "max_tokens"}, "models": [{"id": "Qwen3.5-0.8B-Q8_0", "reasoning": true, "contextWindow": 16384, "maxTokens": 2048}]}`
  (`reasoning`: pi's thinking levels sent as `reasoning_effort`; the streamed `reasoning_content` comes back in the
  history): smoke-tested with Qwen3.5-0.8B Q8_0 and pi 1.4.0 (`pi --mode json --tools read,bash,edit` in a scratch
  directory, a temporary `PI_CODING_AGENT_DIR`): `read`, `bash` and `edit` calls (`edit`'s nested `edits` array
  included) parsed, executed and answered, sessions of up to ~45 tool turns with thinking, each turn prefilling only its
  new tokens; the 0.8B itself is weak at the tasks (wrong paths, loops: it did not finish the edit task).
  `stream_options.include_usage`, `max_tokens` / `max_completion_tokens` (the latter wins) and `cache_prompt` work as
  for other requests. Long sessions: below, Agent sessions.
- **Prefix reuse**: a new request takes the idle slot with the longest common token prefix, truncated
  to it, and prefills only the rest (if that saves at least 32 tokens or half the prompt; else the least
  recently used idle slot). If a running request's slot shares a longer prefix, it is copied
  (`rq_seq_copy`, ~0.6-1.6 ms for LiteLLM's 253-token rubric vs ~110 ms to prefill it); if a request still
  being prefilled will share it, the new one waits for that part and copies it then (a burst of
  classifications prefills the rubric once). Logits are bit-identical to a fresh prefill (tested). A second
  turn re-prefills only the new message; LiteLLM's classifier rubric stays cached. **Slot affinity**: an idle slot
  that would lose more tokens than it keeps (it shares only a prefix with the request, e.g. the system prompt with
  another session or with the first request after a compaction) is not cut: the prefix is copied into the least
  recently used other idle slot, so the conversation it holds finds it whole at its next request.
  **Qwen3.5 / 3.8** (the recurrent state cannot be rewound: a slot can only go back to a checkpoint): each request
  asks for checkpoints where its conversation's next requests leave the cache, in this order of priority within
  `--checkpoints` (default 4): its assistant header (a reply sent back differently from what was generated), the
  end of the system block (the system prompt and the tools: every request of the agent shares it, also the first
  one after a compaction, and a new session), Qwen3.5's turn start (its template drops the reasoning of a turn's
  replies at the next user query, so that request leaves the cache at the turn's first reply), then as the reply is
  generated the position after its `</think>` (content or calls that come back tokenized differently roll back
  there, not to the header) and its end (a retry or another branch after the reply). The slot's other checkpoints
  are dropped first, so new ones never replace these. Measured with pi's sessions below (Agent sessions).
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

## Speculative decoding (TODO S8, G7)

`runq serve MODEL --draft DRAFT` (or `--lookup N` alone): the decoding requests' next tokens are guessed (by the draft
model, or copied from the context by the prompt lookup) and verified in the same batched passes as the other requests'
rows; every output is exactly plain decoding's: the tokens, seeded sampling (each request's RNG draws as plain decoding's),
logprobs (the verify rows' logits are bitwise the decode steps'), streaming, stop strings, `max_tokens`, the end of the
context, reasoning and tool calls, a request batched or alone, speculating or not. Only the speed changes.

**In the engine** (`src/engine.c`, header comment): a decoding request's segment in the pass is `[t, g1..gn]` (t its
sampled token, n guesses) with every row's logits; the rows are sampled in rounds over the requests (one row per
request and round, `rq_sample_batch`), each token through the plain path (output, stops, limits, tool-call scanning);
while the token equals the next row's guess that row is valid, else the sequence is cut after the row it came from
(Qwen3.5 / 3.8: on the row's checkpoint, `rq_seq_checkpoint_rows`). The draft model has a sequence per slot that follows
the slot's tokens (prefix reuse and copies included) and its own context; its passes are batched too: the first one
with each guessing request's catch-up rows and `t` (its first guess from the last row), the next ones with one row per
request still guessing. Its catch-up rows ride along those passes; a pass of its own only for a lag of half the prompt
budget (a prompt, a spell of lookup or plain steps), while the model is in use. Qwen3.5 / 3.8: the slot's
`--checkpoints` stay as G5 plans them; a speculating request gets k + 1 transient checkpoint slots for its verify rows
(dropped at its end, so the next request of the conversation finds what plain decoding leaves; a `</think>` inside a
verify keeps that row's checkpoint as the reasoning's end); the draft mirrors the slot's planned checkpoints. Not for
constrained (`json_schema`) or `force_reply` requests (plain).

**The cost model** (G7, `RqPassCost` in `src/runq.h`): a pass of R rows costs `t_w f(R) + att S1 + att_x S2` (t_w: a
decode step's weights part; f(R): its growth with the rows, f(1) = 1; att: per position of context for the first row of
a segment; att_x: per position for each further row, which shares the first one's read of the KV cache), for the target
and for the draft, calibrated at the start (`--spec-calibrate`, the fastest of 5 runs per configuration) or estimated
from the models' sizes, then corrected online: every decode / verify pass and every draft guess pass is timed against
the calibrated model's prediction, and a moving average of the ratio (per model, the last ~20 passes, 0.5-4x) scales
the costs the policy uses (the load of a shared machine, what the calibration missed). Acceptance: per guess, per request, shrunk to the server's recent counts, shrunk to priors (the
model's 0.6; the lookup's by match length, 0.45 for 2 tokens to 0.94 for 16, from pi's sessions).

**The auto policy** (per pass): the choices (plain, lookup n, model k for each decoding request) that maximize the
decoding requests' expected tokens per second of the pass, greedily: a choice is taken only while it raises the rate,
i.e. its expected gain is > 1 against the batch's own efficiency (a plain batched pass of N requests is already N
tokens for little more than one step). So: at short context the draft model (k 2-3 when its guesses are accepted, the
lookup instead when it matches long enough), at long context the lookup and few or no model guesses (each verify row
pays the attention again), as the batch fills less and then no speculation, and plain steps whenever nothing pays.
A request's draft is kept in sync only if a model pass beats plain for it alone and the time it would save over the
reply's remaining tokens is worth twice the catch-up (a long prompt for a short reply is not). The startup line shows
the cost model and the choice for one sequence alone by position. `--spec fixed` (or a request's `"mode": "fixed"`)
speculates as asked (the CLI's semantics), whatever the batch.

**Per request**: `"speculative": false` (plain), `true` / `"auto"`, `"fixed"`, or an object: `{"mode": "auto"|"fixed"|"off",
"k": 0..16, "tau": 0..1, "lookup": 0..16, "lookup_min": 0 or 2..16}` (k > 0 needs `--draft`: else a 400; without any
source configured, `true` gets the lookup at 3). `timings.speculative`: `{"mode", "passes" (decode passes), "model":
{"passes", "drafted", "accepted"}, "lookup": {...}, "draft_rows" (the draft's rows for this request, catch-up included),
"refed" (positions fed again after a rollback: 0, every rollback lands on its row's checkpoint)}`.

**Tests** (`tests/server_test.py --spec DRAFT|none`, in `make test`'s spec group: the Qwen3-0.6B Q8_0 + Q4_K_M pair,
Qwen3.5-0.8B Q8_0 + Q4_K_M, the 0.8B with the lookup alone; 19 / 22 / 22 checks): every request with speculation off
and with fixed settings (k 1 / 3 / 8 with tau, the lookup at 1 / 4 / 16, both) and auto, compared: content, reasoning,
tool calls, finish_reason, usage and logprobs, streamed and not; greedy, seeded (top-p, top-k), thinking cut inside a
verify pass, `max_tokens` 1 / 2 / 5, stop strings, the end of the context inside a pass, a reply that copies its prompt;
four requests at once; a conversation's next turn reuses the first prompt + its reply (and == a fresh prefill); tool
calls streamed and with thinking (Qwen3.5); no position fed again.

**Measured** (`scripts/bench_serve.py --only gen`, streamed 128-token greedy replies, 2 per client, `--parallel 4`, 8
threads, the bench lock; the shared box at load 5-15 from other jobs: the runs' spread is ~±15%; tok/s in aggregate):

| target + draft, prompts | clients | plain | lookup 3, fixed | draft k 3, fixed | **auto** (draft + lookup 3) |
|---|---|---|---|---|---|
| Qwen3-1.7B Q8_0 + 0.6B Q4_K_M, code | 1 | 62-66 | 82-84 (1.09 tok/pass) | 69-81 (2.49) | **84** (1.8-2.1) |
| | 4 | 190-214 | 203-210 (1.06) | 163-164 (2.56) | **187-208** (1.06: 3-4 model guesses, ~50 / 150 lookup) |
| the same, edit (before the online correction) | 1 | 70-74 | 68-69 (1.05) | 80-96 (3.10) | **84-86** (2.4-3.1) |
| | 4 | 195-213 | 181-199 (1.15) | 202-206 (3.30) | **194-201** (1.12-1.14) |
| Qwen3.5-4B Q8_0 + 0.8B Q4_K_M, code | 1 | 28-31 | 30 (1.03) | 36 (2.70) | **35-37** (2.4) |
| | 4 | 75-78 | 70-71 (1.06) | 63-64 (2.63) | **71-76** (1.02-1.04) |

One client: auto 1.2-1.35x plain, with the draft's guesses where they pay (k 1-3 by the costs) and the lookup's when it
matches; the fixed draft at k 3 is as good on the 4B (the draft costs 0.19 of a step there) and worse on the 1.7B (0.32).
Four clients: a batched pass of 4 requests is already ~3x a step's tokens for ~1.3x its time, so a draft's guesses cost
more than they save (the fixed draft is 15-20% slower there): auto keeps to plain steps and the lookup's occasional
guesses, within the runs' noise of plain (4B, 3 more alternating runs at 4 clients: plain 72-77, auto 73-75). Before the
online correction a noisy start-up calibration once made auto choose the draft in the full batch on the 4B (70 vs 78
tok/s); the passes' measured times now correct the model (`[engine]` summary with `-v`: the measured / calibrated ratio).
`"edit"` asks for a function back with changes: the replies rewrite more than they copy (the lookup's guesses are mostly
wrong; on verbatim copies it is 2.8-3.9x, docs/qwen38-speculative.md, Prompt lookup).

At long context (TODO G7, `build/bench_long -D`, the choice from the measured passes; the 27B extrapolated from
synth-27b: its short-context costs + 16 x the synth's attention): the 27B with the 0.8B Q4_K_M draft (its guesses accepted at p 0.6):
the model at k 2 up to 32k (x1.56 at 512 down to x1.23), k 1 at 64-98k (x1.17 -> x1.13; nothing for p < ~0.45), and
with a 4-token lookup match the lookup, 4 guesses at 512 down to 2 at 98k (x2.15 -> x1.38)
([qwen38-speculative.md](qwen38-speculative.md#in-the-server-s8-and-the-choice-by-position-g7)).

## Agent sessions (pi): the replay benchmark, checkpoints, compaction

**The tool** (TODO G0 (a)): `scripts/replay_sessions.py --model GGUF --ctx N --session FILE | --pick N` replays pi's
recorded sessions (`~/.pi/agent/sessions/*/*.jsonl`: read locally, never copied; the results are numbers only, under
`/var/tmp/runq-replay`) against a fresh `runq serve`, request by request, as pi sends them: `scripts/replay_pi.mjs` runs
pi's own code from its install (pi-coding-agent's session context and `convertToLlm`, pi-ai's `openai-completions`
provider: the same request body, the same SSE parsing into pi's assistant message), with the provider entry above and
the server's context as the model's window, so pi's compaction triggers where it would (`shouldCompact`,
`prepareCompaction`, `compact`: the summary requests go to the server too). `--mode force` (default): each reply is
the recorded one, forced (`force_reply`: the recorded message as the template renders it, fed as if generated), and
the next request carries the server's reply as pi parsed it: the cache sees what a model writing in the template's
format would have generated, and the round trip (our parse, pi's message, pi's serialization, our rendering) is the
real one. `--mode live`: the model generates and its own replies go on (the recorded tool results follow them).
`--probe-every N --probe-tokens T` (force mode): before every Nth tool turn, the same request live (kind `probe`, at most
T tokens, its reply dropped; the forced request after it reuses the prompt from the header's checkpoint): the decode
rate and the speculation at real positions of the session while the rest is replayed (forced replies do not
speculate). `--per-request` prints each request's prefill / decode seconds too. Per
request: kind, prompt / cached / prefilled tokens, "new" (a tool turn's prompt minus the previous prompt and reply:
what a perfect cache prefills), whether the prompt left the slot's tokens before their end (diverged) and where
(`--server-verbose`: the tokens around it), rollbacks, times. Run on Qwen3.5-0.8B Q8_0, 8 threads, 3 slots, 2026-10-28,
the box loaded (prefill seconds noisy, token counts exact), before = `54c2d20` (the replay's hooks only), after = this
work:

| session, context | requests | tool turns: prefilled / new tokens, exact | user turns prefilled | post-compaction prefilled | prefill s |
|---|---|---|---|---|---|
| S1 (71 replies, 46 with reasoning), 32k, before | 98 (11 compactions) | 33,690 / 29,759, 34 of 51 | 117,152 of 156,513 | 214,905 of 214,905 | 1,099 |
| S1, 32k, after | 98 | the same (the 17 others: the last reply, below) | 50,875 | 130,508 | 587 |
| S2 (138 replies, 136 with reasoning), 32k, before (80 requests) | 80 (12 compactions) | 34,040 / 34,035, 36 of 37 | 47,901 of 47,901 | 219,435 of 219,435 | 524 |
| S2, 32k, after | 80 | the same | 28,973 | 134,499 | 370 |
| S1, 96k, before | 74 (1 compaction) | 63,143 / 59,106, 42 of 60 | 354,771 of 498,002 | 32,838 of 32,838 | 1,773 |
| S1, 96k, after | 74 | the same | 129,501 | 25,363 | 962 |

(Tool turns: every one prefilled only its new tokens, or those plus the previous reply and its header: none more,
before or after. Summary requests: the same before and after, ~4-5k tokens each at 32k, 17.5k at 96k, nothing to
share. At 96k, all requests: 519k -> 286k tokens prefilled, the replay's wall time 1,951 -> 1,141 s; the user turns
include a 71k-token re-prefill after pi added a tool, below.)

**Where a turn re-prefilled, and why** (each case from the server's divergence lines):
- **The reply came back tokenized differently** (the last reply re-prefilled, from the checkpoint before the
  difference): (1) an empty reasoning: the generation prompt ends `<think>\n` (that `\n` alone), the model writes `\n`
  `</think>`, and the history renders `<think>\n\n</think>`, whose `\n\n` is one token (14 of S1's 17 at 32k: its
  replies recorded without reasoning; a real model rarely thinks for zero tokens; the 3 others: a reply pi dropped,
  Qwen3.5's reasoning drop); (2) a sampled token where the text's canonical
  tokenization has another (`--mode live` at the model's sampling defaults, 23 replies of a session: ` ren`+`ov` for
  ` renov`, `/s`+`witch`, `Man`+`ual`: 4 of 21 tool turns, 824 extra tokens), in the reasoning or in a call. The
  template cannot make either identical: the text is the same, and the prompt is the canonical tokenization of the
  template's text (tested == HF's ids); reusing the generated tokens for that span would make a request's output
  depend on what was cached (TODO, G5). The `</think>` checkpoint bounds case (2) to the content and calls when the
  difference is after the reasoning (`tools/ckpt_think`). Not seen: reasoning or content re-serialized differently
  (`reasoning_content` round-trips: pi sends back what it got, joined with `\n`; runq trims as the template does),
  tool-call arguments formatted differently (the template renders the parsed values, not pi's JSON text).
- **Qwen3.5's template drops a turn's reasoning at the next user query** (Qwen3.8 keeps it): that request leaves the
  cache at the turn's first reply. Before: no checkpoint below it but the header's, so back to 0; after: the turn's
  checkpoint (user turns 117k -> 51k tokens on S1).
- **Compaction**: the summary request shares nothing (pi's summarization system prompt + the serialized
  conversation); the next request is the system block + the summary + the kept messages: before, nothing reused (the
  slot's checkpoints were above the system block); after, the system block (7-10k tokens with pi's prompt and tools),
  from the session's slot, which the summary request did not take (slot affinity, >= 2 slots). Two of S2's 12 found a
  different system block (pi rebuilds it from the compaction entry's checkpoint, without an update made since).
- **pi changed its system prompt or tools mid-session** (a system message in the session: S1 added a tool after 36
  turns): the system block is rendered at the top, so everything is prefilled again (71k tokens at 96k: ~5 min on the
  0.8B, which pi's default HTTP idle timeout of 300 s ended before the keep-alive comments). Inherent to the template
  (no system message after the first); the same with a thinking-level change on Qwen3.8 (its effort instruction
  opens the system block).
- **An aborted reply** (pi drops it from the history): back to that reply's header.

**For the 27B agent setup**: `runq serve ... --timeout 0` (a compaction's next request prefills for ~16-19 min on the
27B at 8 threads, past the default 600 s); the SSE keep-alive comments keep pi's `httpIdleTimeoutMs` (300 s) from
ending long prefills; two slots if the memory allows them (`--kv-mem`: the summary request then does not take the
session's slot, and the next request reuses the system block; with one slot it is prefilled again: ~8k tokens, ~3
min); `--checkpoints 3` saves 150 MiB per slot on Qwen3.8 (its header, the system block and the `</think>` or the
reply's end; Qwen3.5's turn checkpoint is not needed there). pi's compaction settings and a compaction's cost on the
27B: [qwen38-speculative.md](qwen38-speculative.md#agent-sessions-replay-and-compaction-g0-a-g5). The configuration
measured on the 27B: next section.

## Qwen3.8-27B with pi

The setup for agentic development at 96k on the 16-vCPU / 31 GB box (2026-10-28; TODO's "Agentic development";
measurements in [qwen38-speculative.md](qwen38-speculative.md): G3, "Long-context costs", "Agent sessions"), with the
box's other services stopped (~24 GiB available):

```sh
OMP_NUM_THREADS=16 runq serve models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_M.gguf -c 98304 -k bf16-dot \
    --draft models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --slots 1 --parallel 1 --checkpoints 3 \
    --spec-calibrate 32768 --timeout 0 --name qwen3.8-27b --port 18080
```

- **The file: Q3_K_M** (`scripts/download_models.py 3.8-27b-q3km`). At `-c 98304` runq's estimate is 19.45 GiB (weights
  12.40, a 96k sequence: KV 6.0 + the recurrent state and 3 checkpoints), + the draft at that context (2.03 GiB: its
  sequence 1.23) and 0.66 GiB for the verify rows' checkpoints: ~22.1 GiB, ~2 GiB to spare. The Q4_K_M needs ~25 GiB
  there (it fits at 96k only without the draft and with ~1 GiB to spare, or with the draft up to ~64k). The price:
  the Q3_K_M disagrees with the API's model on ~3.5% / 5.5% of the tokens (thinking off / on; Q4_K_M 1.6% / 3.1%),
  +2.7% perplexity, KL 0.03 from the Q4_K_M on agentic replies (G3).
- **`-k bf16-dot`**: the attention's dot products in bf16 (the KV cache is as big as f16's): at 98k a 2,560-token
  chunk prefills in 86 s instead of 123 (16 threads), decode 4-7% faster; within f16's noise on Qwen3.5 / 3.8 (KL), and
  the 27B's retrieval of a fact ~84k tokens back passes with it (TODO G2).
- **16 threads** (all the vCPUs; 8 leave room for other work at ~1.4-1.7x the time: decode at 98k 5.7 vs 6.9 tok/s, the
  chunk 148 vs 86 s).
- **Speculation**: `--draft` the 0.8B Q4_K_M with the default auto policy (and the lookup at 3 guesses): at short context
  the draft (the live session below: 15-23 tok/s at 2-3k positions), at long context mostly the lookup (each verify
  row pays the attention again: the replay's probes 8.1-13.9 tok/s at 21-81k, ~7-9 plain). `--spec-calibrate 32768`:
  at the default 8,192 positions the 27B's attention slope is within the noise of its ~90 ms step; even at 32,768 a
  noisy box can make the calibration fall back to its estimate (2.40 us per position, ~4x the real ~0.6: printed on
  the startup line; the online correction rescales the whole cost, TODO G7 (b)): restart the server if it shows 2.403.
- **One slot** (`--slots 1 --parallel 1`): a second 96k slot (7.8 GiB + the draft's 1.2) does not fit, so a
  compaction's summary request takes the session's slot and the next request prefills the system block again (~2 min;
  TODO G5 (b)). `--checkpoints 3`: Qwen3.8 needs the header, the system block's end and the reply's end. `--timeout 0`:
  a compaction's requests prefill for 4-8 min each; the SSE keep-alive comments keep pi's idle timeout from firing.

**pi** (1.4.0; a temporary agent directory, `PI_CODING_AGENT_DIR=DIR`, so `~/.pi` is untouched): `DIR/models.json`

```json
{"providers": {"runq": {"baseUrl": "http://127.0.0.1:18080/v1", "api": "openai-completions", "apiKey": "none",
  "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": true, "supportsStore": false, "maxTokensField": "max_tokens"},
  "models": [{"id": "qwen3.8-27b", "name": "Qwen3.8-27B (runq)", "reasoning": true, "input": ["text"],
              "contextWindow": 98304, "maxTokens": 16384, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}
```

and `DIR/settings.json` `{"defaultProvider": "runq", "defaultModel": "qwen3.8-27b", "defaultThinkingLevel": "medium",
"compaction": {"enabled": true, "reserveTokens": 16384, "keepRecentTokens": 8000}}`: `keepRecentTokens` 8,000 instead
of pi's 20,000 makes a compaction's next request ~12k tokens shorter (~4-7 min less on the 27B); `reserveTokens` 16,384
(pi's default) compacts at ~82k. `maxTokens` matters for speculation: the server follows a request with the draft only
if the time it saves over the remaining `max_tokens` is worth the draft's catch-up. Thinking `medium` sends no effort
instruction (Qwen3.8's `xhigh`, its default, reasons longest).

**Measured** (the Q3_K_M, the command above; the box at load 10-20 from other jobs):
- **A live session** (`pi -p --mode json --tools read,bash,edit,write --no-extensions --no-skills --no-context-files`
  in a scratch Python project: "add `median(xs)` to `textstats/stats.py` (ValueError on an empty list, the mean of the
  middle values for an even length), tests for it, run them with `python3 -m unittest -v`"): **done in 6 requests, ~80
  s**: `ls` + `cat` of the files, an `edit` of the module, a `cat -A` check, an `edit` adding 4 tests, the test run (7 / 7
  pass), a summary; the code correct and in the file's style. Per request: the first (1,694 tokens: pi's system prompt
  with 4 tools) prefilled in 24 s, the others 0.6-5 s (29-354 new tokens), replies of 46-245 tokens at **15-23 tok/s**
  (thinking medium: short reasoning).
- **A replayed session** (54 replies, 144k tokens, compacted once at 84k; replies forced to the recorded ones, live
  probes every 4th turn): every tool turn prefills only its new tokens (P / 67.5 + 0.19 us x P x (L + P / 2) seconds:
  e.g. 2,600 tokens at 72k in 70 s), decode with speculation 8-14 tok/s at 21-81k, a compaction ~12.4 min (a
  30k-token summary request 8 min, the next request 4 min).
- **What to expect per turn** (2,500 new tokens + a 500-token reply): ~1.2-1.4 min at 8k, ~1.5-1.8 min at 32k, ~2.1-2.5
  min at 90k (the lower bounds with speculation's x1.2-1.5 on the reply); a compaction ~12 min (~5 turns' worth).

## Limitations

- Output is constrained only to JSON schemas with a finite set of outputs (no `json_object`, no free strings);
  tool calls only with the Qwen3.5 / Qwen3.8 templates, `tool_choice` auto / none, tool parameters not constrained
  while decoding; no images, `n > 1`; penalties are ignored.
- No TLS: bind it to localhost or a trusted network (`--api-key` is a shared secret over plain HTTP).
- The KV memory is a fixed pool (`--slots` x `-c`, no paging or preemption): a request keeps its slot until it ends.
  With `--draft`, each slot also holds a draft sequence of `-c` positions (counted in the pool).
- One model per process (and its draft). POSIX only (on Windows the server reports that it is not available).

