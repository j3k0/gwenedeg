# Gwenedeg

Gwenedeg runs dense Qwen3, Qwen3.5 and Qwen3.8 language models on the CPU, in plain C with no third-party libraries
(libc, libm, pthreads). It is a small inference library (`src/runq.h`), a command line, and an OpenAI-compatible HTTP
server. **The binary is called `runq`**, as in [qwen3.c](https://github.com/adriancable/qwen3.c), the project it grew
from; the library keeps the name too (`runq.h`, `rq_` prefix). It loads GGUF files (Q8_0, the K-quants, Q2_K and the
i-quants IQ1_S .. IQ4_XS) directly, weights, tokenizer and hyperparameters from the file.

## What is different

- **Bit-exact determinism.** Logits are bit-identical whatever the thread count, the prefill chunk size or the batch
  composition; a batched decode row equals decoding that sequence alone; the AVX-512, NEON and generic C kernels give
  the same bits, because every kernel follows one written arithmetic definition. The test suite checks all of it
  (against Hugging Face transformers goldens as well: [tests/README.md](tests/README.md)).
- **i-quant kernels for AVX-512.** Q2_K and all of IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS, on
  weights repacked at load, with an optional lossless 4-bit repacking of IQ3 weights (`--repack`). Underdog Saluki 27B
  (a 7.9 GB Qwen3.8-27B mix of these types) runs in 8.4 GB of RAM. I-quant prefill is ~0.72x the K-quants' on AVX-512
  (measured, [docs/iquants.md](docs/iquants.md)).
- **Speculative decoding that does not change the output.** A draft model of the same vocabulary, the model's own MTP
  block (multi-token prediction: Qwen3.8-27B's `blk.64`), or a prompt lookup guesses; the model verifies the guesses in
  one batched pass. The output is exactly plain decoding's, greedy or seeded, whatever the draft. In `runq serve` the
  guesses of all running requests are batched into the same passes (continuous batching), and an auto policy chooses the
  source and the guess count per pass from a cost model ([docs/server.md](docs/server.md#speculative-decoding-todo-s8-g7)).
- **KV cache types.** f16 (default), bf16 and f32; `bf16-dot`, the attention's dot products as the AVX512_BF16
  instruction computes them; and 8-bit / 8-bit K + 4-bit V / 4-bit Walsh-Hadamard-rotated caches (`q8`, `q4`, `q4k4`:
  0.53 / 0.41 / 0.28 of f16's memory; Qwen3.8-27B at 98k positions: 6.0 GiB of KV in f16, 3.19 / 2.44 / 1.69 GiB). All
  are bit-exact across batches and threads. Their accuracy cost is measured per type:
  [docs/usage.md](docs/usage.md#kv-cache-types), [docs/kv-quant.md](docs/kv-quant.md).
- **OpenAI-compatible server** (`runq serve`): chat completions with streaming, continuous batching (the running
  requests, 16 by default, share each forward pass), KV prefix reuse across requests, tool calls for Qwen3.5 / 3.8, constrained
  decoding for JSON schemas with a finite set of outputs, memory-aware admission, and an optional **disk tier** for the
  KV cache that survives eviction and restart (a 27k-token Qwen3.8-27B Q4_K_M conversation, 16 threads: 408-418 s to prefill, 0.58-0.61 s to
  load back; 2026-10-31, [docs/kv-disk.md](docs/kv-disk.md)).
- **Exact tokenizer.** Byte-level BPE matching Hugging Face `tokenizers` on Qwen's tokenizer, including NFC
  normalization (tested against HF).

## Headline numbers

Every number below says when, where and how it was measured; the full tables, setups and caveats are in
[BENCHMARKS.md](BENCHMARKS.md). The machine is a 16-vCPU AMD EPYC Genoa (Zen 4) KVM VM with 30 GB of RAM, shared with
other workloads (run-to-run noise ±15-30%).

**Qwen3.8-27B against ik_llama.cpp** (2026-11-21, one window, the same IQ3_S file of 12.1 GB, 14 threads, an 8-bit K
/ 4-bit V cache in both, MTP speculative decoding in both, the same server benchmark `scripts/s9c_bench.py`;
[BENCHMARKS.md](BENCHMARKS.md#qwen38-27b-iq3_s-vs-ik_llamacpp-server-workloads-2026-11-21)):

| tok/s unless noted | runq serve | ik_llama.cpp, 1 slot + MTP (our tuning) |
|---|---|---|
| decode, 1 client, thinking on / off | **18.6 / 21.8** | 7.8 / 11.2 |
| decode, 4 clients at once (aggregate), thinking on / off | **23.4 / 26.4** | 6.6 / 8.5 (MTP needs a single slot) |
| decode in a 32-37k-token session | **13-16** | 2-2.4 |
| a pi-shaped follow-up turn (+2.5k tokens) on a 32k session, wall | **76-81 s** | 301-316 s |
| prefill, 0.8-2.5k-token prompts | 81-84 | **110-118** |
| prefill, a cold 31.8k-token prompt | 471 s | **319 s** |
| perplexity on the same tokens (prose / code) | 9.483 / 7.133 | 9.493 / 7.145 |

Same quality (the perplexities differ by 0.1-0.2%, within the standard error). Per cell, runq decodes 1.9-2.4x faster at
one client, 3.1-4.0x at four clients (against ik's single MTP slot) and 5.5-8x faster in the 32k session; ik_llama.cpp
prefills 1.35-1.5x faster. One run per cell on a shared VM; ik_llama.cpp was configured and tuned by us from its
documentation, and its authors may know better settings; the KV caches have equal bit widths but are not the same
quantizer. Other engines (mainline llama.cpp, vLLM's CPU backend) and other CPUs were not measured.

Smaller models, same VM (2026-10-26, best of 2, 8 / 12 threads): Qwen3-0.6B Q8_0 prefills 1,813 / 2,287 tok/s and decodes
207 / 165; its Q4_K_M 1,721 / 2,338 and 278 / 373; Qwen3.5-4B Q8_0 308 / 443 and 31 / 30; Qwen3.8-27B Q4_K_M 45 / 65 and
8.5 / 10.0 (all-models table, perplexities and caveats:
[BENCHMARKS.md](BENCHMARKS.md#all-model-files-on-the-epyc-vm-2026-10-26-to-2026-10-30)). Against upstream qwen3.c on
Qwen3-0.6B Q8_0 (2026-10-25, one session): prefill of 451 tokens 2,058 tok/s at 8 threads, where qwen3.c prefills token by
token at ~99; decode 148 against 101. Apple M1 Pro (NEON) numbers are in BENCHMARKS.md too.

## Quick start

Requirements: `make` and gcc (Linux) or Xcode's command-line tools (macOS); Python 3 (standard library only) for the
model download script.

```sh
make                                      # -> ./runq
scripts/download_models.py q8gguf         # Qwen3-0.6B Q8_0 (639 MB), pinned revision, checksummed
G=models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf

./runq $G -c 4096                         # interactive chat: one message per line, empty line / EOF ends
./runq $G -c 4096 -t 0 -n 200 -i "Why is the sky blue?"     # one turn, greedy, at most 200 tokens
./runq serve $G                           # OpenAI-compatible server on http://127.0.0.1:8080/v1

curl -s localhost:8080/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hi!"}], "max_tokens": 32}'
```

`./runq` without arguments prints every option; [docs/usage.md](docs/usage.md) explains them. The server has no TLS
and binds to 127.0.0.1 without authentication by default (`--api-key KEY` requires a bearer token): keep it on localhost
or a trusted network.

## Models

Qwen3 dense (0.6B, 1.7B, 4B, 8B; 4B-Instruct-2507 and 4B-Thinking-2507), Qwen3.5 dense (0.8B and 4B tested; the hybrid
Gated DeltaNet + gated attention architecture) and Qwen3.8-27B, including Underdog Saluki 27B. GGUF types: Q8_0,
Q3_K_M, Q4_K_M, Q5_K_M, Q2_K and the i-quant mixes (any mix of Q8_0, Q2_K .. Q6_K and IQ1_S .. IQ4_XS tensors), plus
qwen3.c's Q8_0 `.bin` export. `scripts/download_models.py --list` shows the downloadable sets. The model table, the
chat templates and what is not supported (MoE, images, RoPE scaling, other GGUF types) are in
[docs/models.md](docs/models.md). No model weights are in this repository.

## Documentation

| | |
|---|---|
| **Users** | [docs/usage.md](docs/usage.md): command line, memory check, chat, speculative decoding, KV cache types · [docs/models.md](docs/models.md): model sets and files |
| **Operators** | [docs/server.md](docs/server.md): `runq serve` flags, endpoints, tool calls, speculation, pi / LiteLLM setups · [docs/memory-admission.md](docs/memory-admission.md): `--admit` · [docs/kv-disk.md](docs/kv-disk.md): the disk tier |
| **Internals** | [docs/library.md](docs/library.md): the C API and source map · [docs/iquants.md](docs/iquants.md) · [docs/kv-quant.md](docs/kv-quant.md) · [docs/qwen35.md](docs/qwen35.md) · [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): changes from qwen3.c, rejected experiments, CI |
| **Measurements** | [BENCHMARKS.md](BENCHMARKS.md), [bench/](bench/) (raw logs) |
| **Open work** | [TODO.md](TODO.md), [IDEAS.md](IDEAS.md) |
| **History and lab notes** | [docs/notes/](docs/notes/qwen38-speculative.md): the dated notebook of the Qwen3.8-27B speculative-decoding work |
| **Tests** | [tests/README.md](tests/README.md) |

[docs/README.md](docs/README.md) is the index.

## Limitations

- Linux (gcc) is the main platform. macOS on Apple Silicon (Apple clang) builds and runs with the NEON kernels (tested on macOS only; the
  tests that compare generated text with goldens recorded on x86 need an arm64 baseline: [docs/usage.md](docs/usage.md#limitations)). The
  Windows code paths inherited from qwen3.c are kept but untested, and the server is POSIX only.
- Fast paths: x86-64 with AVX-512 VNNI (selected at compile time with `-march=native`) and AArch64 with the dot-product
  extension. Other CPUs, including x86 without AVX-512 VNNI, use the generic C kernels: the same results, but for the
  K-quant files ~10x slower prefill and ~3-4x slower decode. CPU only; one model per process (and its draft).
- Dense models only: no MoE, no images, no RoPE scaling, no Jinja interpreter (the chat templates of Qwen3, Qwen3.5 and
  Qwen3.8 are built in). `runq serve`: no TLS, `n` = 1, tool calls only with the Qwen3.5 / 3.8 templates, a fixed pool of
  KV slots.

The full list is in [docs/usage.md](docs/usage.md#limitations) and [docs/server.md](docs/server.md#limitations).

## Testing

`make test` builds the binaries and runs the regression suite (tokenizer ids and logits against Hugging Face, batched
against sequential, thread-count and SIMD-against-generic bit-identity, the KV cache types, speculative decoding against
plain decoding, the server over HTTP, synthetic models with the bigger models' shapes); it needs the Qwen3-0.6B files
and numpy. `make test-models` runs every downloaded bigger model. Counts, timings and the other test groups:
[tests/README.md](tests/README.md); CI with Dagger and the benchmark scripts: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Credits and license

Gwenedeg is MIT-licensed ([LICENSE](LICENSE)). The engine started from [qwen3.c](https://github.com/adriancable/qwen3.c)
by Adrian Cable (MIT), itself based on Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) (MIT); an
unmodified copy of qwen3.c is kept in [third_party/qwen3.c](third_party/qwen3.c). The GGUF format and the quantization
block formats follow [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT); the implementation here is our own.
The Qwen models are by the Qwen team (Alibaba Cloud), Apache-2.0; **no model weights are included**,
`scripts/download_models.py` fetches them from Hugging Face. Third-party code, data and formats (Unicode tables, Qwen
chat templates and test data, LiteLLM's classifier rubric, the Beagle perplexity text, the optional mynah-slm
comparison) and their licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
