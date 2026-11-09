# Memory-aware admission

Status: implemented (2026-11-07), but the measurements (below: they need the 27B model) and the decision on the
default, which stays `worst` until then (TODO.md, Server). Steps 1-3 admit a request on the memory it can ever use and
never take memory back from a running one. Step 4 reserves memory in chunks as a request generates, with pausing and
swapping out to disk as the guaranteed fallback. Steps 1-3 are written so that step 4 only sets a finite chunk and adds
the renewal and the swap. Each section's "As built" says where the code departs from the plan.

What exists:
- **Library** (runq.h, src/forward.c): `RqSeq.hwm`, `rq_seq_release`, `rq_seq_resident`, `rq_seq_kv_bytes`.
- **`--admit worst`** (the default): today's pool, unchanged but for the calibration's release.
- **`--admit bound`** (steps 1-3): leases of prompt + `max_tokens`, `committed` within `--kv-mem`, the LRU reclaim of
  idle slots (saved first with `--kv-disk`), strict FIFO waits, the live budget (MemAvailable - `--admit-margin` -
  the running jobs' promised growth), the guess cap, `/health`'s `mem`, the timings `mem_wait_ms`, `lease_gib`.
- **`--admit chunk`** (step 4; needs `--kv-disk`): leases of prompt + `--admit-chunk` tokens renewed chunk by chunk,
  parking, the oldest job's precedence, the swap-out / swap-in to `DIR/<model>-<KV type>/swap/`, the timings
  `renewals`, `paused_ms`, `swaps`, `swap_ms`, `yields`; **`--admit-yield N`**: the round-robin of the compute seats.
- **Tests:** api_test's `seq_release` (both builds); server_test's `admit` group (`--admit`: `bound`) and `admit_chunk`
  group (`--admit-chunk`: `chunk`, swaps, a swap mid-prefill, the draft's swap I/O failing, yield), run by
  tests/check.py's server group on Qwen3-0.6B and Qwen3.5-0.8B (with its Q4_K_M as the draft).

Known limitations:
- Pages the OS swapped out still count in `res` (conservative for the static check, not for the live one: step 3).
- Transparent huge pages: a released range can stay resident up to 2 MiB past `kv(hwm)` per (layer, K / V) range
  (step 2), inside the margin, not in `committed`.
- The swap's disk room is checked at the start only: other processes filling the disk make a swap-out (or a swap-in)
  fail, and that request ends with a 500 (no other request is touched).
- A paused (parked, yielded) job keeps its memory until a renewal or an admission needs it; `chunk`'s swap guarantees
  the oldest job, not throughput.

## Goal

Run as many requests at once as the memory they actually use allows, instead of as many as fit if every slot filled
to `-c`.

- **Steps 1-3:** a request's memory is bounded before it starts. The engine never generates past `max_new` (serve.c
  clamps it to `-c - n_prompt`), and speculation's guesses are capped by it too (engine.c, `spec_cands`: `lim <=
  max_new - n_gen - 1`). So `n_prompt + max_new` positions is a hard limit, not a guess, and admitting on it cannot
  overrun the budget. A request that does not fit waits in the queue until running ones finish.
- **Step 4:** a request reserves its prompt plus one chunk of generation (`--admit-chunk`, 1,024 tokens), and renews
  the reservation chunk by chunk. Requests without `max_tokens` (or with a large one) no longer reserve the whole
  context.

Rules that stay: a sequence's logits never depend on the batch, the path or the thread count. Admission, pausing and
swapping only change when a request computes, never what it computes. Released memory reads back as a fresh
sequence's (zeros), which the attention never reads below `len` anyway.

## Today

- `engine_start` sizes the pool for the worst case: `nslot x per_slot <= budget - spec_res`, where `per_slot` is
  `rq_seq_mem` at `-c` (KV, recurrent state x (1 + `--checkpoints`), the draft's sequence) and `spec_res` reserves
  `parallel x (spec_kmax + 1)` state copies for the verify rows. The budget is `--kv-mem`, else half the RAM, at most
  MemAvailable after the load. `--slots` and `--parallel` are lowered to fit.
- The memory itself is anonymous `mmap` (`alloc_big`), touched as used. Saluki at q8 with 2 slots of 98,304:
  25.7 GiB of address space, 8.9 GiB resident (weights ~7.5 GiB).
- Nothing is given back while the process runs. `rq_seq_truncate` only moves `len`; checkpoint buffers are allocated
  at first use and kept "for the next one" (`rq_seq_checkpoint_drop`). The verify rows' transient checkpoints are the
  exception: `spec_release` frees them (`rq_seq_set_checkpoints` back to `--checkpoints`). So each slot keeps the
  memory of the longest sequence it ever held. The speculation calibration also leaves slot 0 touched up to
  `--spec-calibrate` positions (8,192 by default).
- `admit()` returning -1 already means "wait": `engine_main` stops admitting for this step and the job keeps its
  place in the FIFO.

Saluki q8 refused 3 slots because of the default budget, not the RAM: the startup line says "pool budget 15.30 GiB,
half the RAM", and 3 x 4.61 GiB (slots) + 3 x 0.99 GiB (verify rows) = 16.8 GiB.

## The memory model

All terms come from quantities the engine already has. Numbers: Saluki IQ2-mix, `-k q8-dot`, draft Qwen3.5-0.8B
Q4_K_M, `--checkpoints 4`, `-G 5`, from the startup line of the running server (2026-11-07). The drafter is a draft
model (`--draft FILE`) or the MTP view (`--draft mtp`, TODO S9c: the model's own MTP block, one attention layer and no
recurrent state); the last column is any Qwen3.8-27B file with its MTP block (ISTA's IQ3_S-mtp: the terms depend on
the shapes only), derived from the shapes (TODO S9c.4; the 27B's startup line confirms it in S9c.G).

| term | what | Saluki q8, 0.8B draft | Qwen3.8-27B q8, `--draft mtp` |
|---|---|---|---|
| `kv(n)` | KV bytes of positions 0..n-1, target + the drafter (a draft model, or the MTP view: one attention layer, no state): whole blocks of ATT_P = 256 positions, page-rounded per (layer, K / V) range, + the token arrays | 34.8 KB + 6.5 KB = **41.3 KB per position** (10.1 MiB per 256-position block) | 34.8 KB + 2.2 KB = **37.0 KB per position** |
| `st` | one copy of the recurrent state, target + the drafter (the MTP view: none; 0 for Qwen3) | 149.6 + 19.3 = **168.9 MiB** | **149.6 MiB** (the target's) |
| `ck` | `--checkpoints` | 4 |
| `kk(j)` | the job's guesses per pass at most: `max(spec_k, spec_lookup)` if it speculates (`spec_start`'s rule: no constraint, no forced reply), else 0 | 5 |
| `C` | the lease chunk, in generated tokens: unlimited in steps 1-3 (`--admit bound`), `--admit-chunk` in step 4 (`--admit chunk`) | 1,024 |
| `small` | per running job: the lookup index (~1.4 MB at 98k), per slot: mask / logprob buffers (~1.3 MB) | ignored, inside the margin |

**A running job's lease** (the memory reserved for it):

```
P(j)     = min(capacity, n_prompt + max_new)            the positions it can ever write (a hard limit)
E(j)     = min(P(j), n_prompt + C)  at admission         the end of its lease; step 4 renews it to min(P(j), E(j) + C)
lease(j) = kv(E(j)) + st x (1 + ck + (kk(j) > 0 ? kk(j) + 1 : 0))
```

With `C` unlimited, `E = P`: the lease is the job's hard limit, and it never needs renewing. The state copies are
reserved whole from the start (they are allocated lazily, but a job may take them at any point).

| request | P | lease, `C` unlimited (steps 1-3) | lease, `C` = 1,024 (step 4) | reserved today |
|---|---|---|---|---|
| short, 800 + `max_tokens` 1,024 | 2,048 (8 blocks) | 0.08 + 1.81 = **1.89 GiB** | 1.89 GiB | 5.60 GiB |
| pi turn, 60k + `max_tokens` 16,384 | 76,544 | 2.95 + 1.81 = **4.76 GiB** | 2.36 + 1.81 = **4.17 GiB** | 5.60 GiB |
| 800, no `max_tokens` | 98,304 | 3.79 + 1.81 = 5.60 GiB | **1.89 GiB** | 5.60 GiB |
| 60k, no `max_tokens` | 98,304 | 5.60 GiB | **4.17 GiB** | 5.60 GiB |

For short requests ~96% of the lease is state copies (11 x 168.9 MiB = 1.81 GiB), not KV. Each renewal of step 4 adds
`kv(C)`: 41 MiB on Saluki.

With the MTP view instead of the 0.8B draft (the same `ck` and `kk`; 11 x 149.6 MiB = 1.61 GiB of state copies): the
short request's lease is **1.68 GiB** (1.89), the pi turn's **4.24 GiB** (4.76), a 98,304-position job's **4.99 GiB**
(5.60); the drafter's sequence at 98,304 positions ~0.20 GiB of KV and no state (the 0.8B's: ~0.6 GiB + 19.3 MiB per
state copy). Outside the pool (process memory, not leases): the view's weights (ISTA's Q6_K block ~0.33 GiB; the
0.8B's ~0.5), its context (at most the target's runtime) and the hidden-state buffers (21 MB). The formulas do not
change: the engine's terms take the view as the draft (`rq_seq_mem` of the view: KV only), and the startup line's
`admission bound:` KV per position and state are the sums above (tests/server_test.py `--admit --draft mtp`:
`admit/mtp_terms`).

**A slot's resident memory** (running or idle):

```
res(slot) = kv(hwm(seq)) + kv_draft(hwm(dseq)) + st_target x buffers(seq) + st_draft x buffers(dseq)
```

`hwm` is the sequence's high-water mark (the longest length since its last release), `buffers` the state buffers it
holds (the live state once touched, plus every allocated checkpoint buffer). Positions only ever grow from `len`,
so `kv(hwm)` is an upper bound on what is resident (a partial block counts whole).

**The pool's commitment:**

```
committed = sum over slots of: running ? max(res, lease(job)) : res      (KV and state parts each maxed separately)
```

Admissions and renewals keep `committed <= budget`. Idle slots never grow, and a running job never writes past
`E(j)`, so the invariant holds between them. In steps 1-3 that is automatic (`E = P`); in step 4, `step()` renews a
lease before a pass would write past it (below).

## Step 1: admission on the lease

**Where:** `admit()` in src/engine.c, after it has chosen the slot `si` (and copy source `src`, prefix length `L`)
the way it does today, before `dk_go` / `start_run`.

1. `need = committed - res(si) + lease(j)`. The slot's old contents past `E(j)` are released in `start_run` (step 2),
   so they do not count. A copy source is unchanged.
2. If `need <= budget`: admit as today.
3. Else reclaim idle slots, least recently used first, never `si` or `src`, until `need` fits:
   - with `--kv-disk`, and if `dk_wants` the slot: plan its spill (`e->plan.spill`) and return 2. `dk_run` saves it,
     and the job is admitted again at the next iteration, keeping its place. One spill per iteration, as today.
   - then `rq_seq_reset` + `rq_seq_release(seq, 0)` on the slot and its draft sequence. The cache is lost (or on disk).
4. If reclaiming every eligible idle slot is still not enough: return -1. The job waits, and so does everything behind
   it (strict FIFO, as `engine_main` does for -1). It is retried every step; `reap` lowers `committed` when a job
   ends.
5. Never wait when nothing is running (`nrun == 0`): reclaim every idle slot and admit. The startup check guarantees
   that one job of `-c` positions fits the static budget, so this cannot overrun it (step 3's live budget can be
   lower: then admit anyway and log a warning).

Reclaim is computed first and done only if it makes the job fit, so a job that will wait anyway does not destroy
caches.

As built (engine.c, "memory admission"):
- `committed` is not kept up to date: `mem_admit` recomputes it from the slots at each admission (`rq_seq_resident`
  per sequence, ~1 us; a few slots). So `start_run`, `end_run` / `reap`, `dk_load`, a copy, the verify rows' transient
  checkpoints and a reclaim need no bookkeeping of their own; a `Run` keeps its `P`, `E`, `kk` and lease (KV, state).
- A disk load (`dk_best`'s plan) first checks the room for the job on the slot it fills (it may reclaim or wait).
- Slot affinity (copying an idle slot's prefix into another idle slot): if the copy does not fit beside its source
  even with every other idle slot reclaimed, the job takes the source slot itself.
- A job that does not use the draft model leaves the slot's draft sequence as it was (`spec_start`); with `bound`,
  `start_run` truncates it to its common prefix with the prompt if it is longer than `E`, so the slot stays within
  the lease.
- The calibration's release is done in both modes (it only gives memory back: `worst` is otherwise unchanged).
- `-v`: `[engine] job waits for memory: needs 0.44 GiB (16 + 4080 positions), 0.36 free with the idle slots reclaimed
  (budget 0.45, committed 0.08, 3 running)` (once per job), `[engine] slot 0 reclaimed: 120 tokens, 0.03 GiB (saved to
  disk)`.

**Startup (`engine_start`)** with `--admit bound` (and `chunk`):
- `--slots` is no longer lowered to `budget / per_slot`. Default `--slots`: `parallel + 4`, as today.
- The startup refuses only if one worst-case job does not fit: `per_slot + st x (spec_kmax + 1) > budget`. A request's
  `"speculative"` (up to 16 guesses, or the lookup with no `-G` / `--lookup`) is capped in `engine_submit` at the most
  guesses whose state copies one `-c` job holds within the budget (`per_slot + st x (k + 1) <= budget`, so at least
  `spec_kmax`), on Qwen3.5 / 3.8 only (`st > 0`; Qwen3's guesses cost no lease), so the check stays true. Speculation
  changes only the speed.
- `spec_res` is no longer reserved up front: it is part of each speculating job's lease.
- `--parallel` stays the compute cap (per-request speed drops 15 -> 10 -> 8 tok/s from 1 to 3 requests on Saluki).
- After the speculation calibration, `rq_seq_release` slot 0's sequences (gives back up to `--spec-calibrate`
  positions: 0.32 GiB at 8,192, 1.3 GiB at 32,768).

The address space grows with `--slots` (each slot is still `mmap`ed at `-c`): 6 slots on Saluki are ~28 GiB of pool.
That needs `vm.overcommit_memory` 0 or 1 (this box: 0). Under 2 the `mmap` fails at startup: we log it and refuse.

## Step 2: giving memory back

**New library calls** (runq.h, src/forward.c):

```c
// gives back the memory of the KV blocks wholly past max(keep, len) (madvise MADV_DONTNEED; they read as zeros
// afterwards, as a fresh sequence's), and of the checkpoint buffers that hold no checkpoint; len 0: the recurrent
// state's buffer too. The sequence's contents below max(keep, len) are untouched. RQ_OK, RQ_ERR_ARG (keep < 0)
int rq_seq_release(RqSeq *s, int keep);
// the memory the sequence holds (an upper bound): kv = the blocks up to its high-water mark, state = its live state
// (if touched) + allocated checkpoint buffers, total; and the KV bytes n positions would take (for leases)
void rq_seq_resident(const RqSeq *s, RqSeqMem *out);
size_t rq_seq_kv_bytes(const RqSeq *s, int n);
```

**The high-water mark:** a new `RqSeq.hwm`, raised wherever `len` grows: the batched pass (`s->len += sg[i].n`,
which includes verify rows that are later rejected), the single-row decode (`s->len = pos + 1`), `rq_seq_copy`,
`rq_seq_load`, `rq_seq_fill_synthetic`, and the cost calibration (through the same passes). One helper,
`seq_grow(s, n)`, used at each site. `rq_seq_release` lowers it to `max(keep, len)` rounded up to a block.

**The release:** the layout (`kv_tile`) puts the blocks of ATT_P positions of one layer one after the other (the
heads inside a block), so the tail past block `b0 = ceil(max(keep, len) / ATT_P)` is one contiguous range per
(attention layer, K / V). For each: start = the byte offset of block `b0`, rounded **up** to a page; end = the end of
block `ceil(hwm / ATT_P)`, rounded **down**; `madvise(MADV_DONTNEED)` if start < end. On the 27B that is at most 32
calls per sequence. One layer's K or V block is tens of pages (~280 KB on the 27B at q8), and rounding the start up
and the end down means a kept block is never touched. With nothing kept (`max(keep, len) == 0`) the K and V mappings
go whole: 2 calls.

Transparent huge pages: with THP `always` (this box) a fault maps a 2 MiB page, so a range can be resident up to
2 MiB past `kv(hwm)` (seen in api_test: a release to 0 gives back more than the estimate). Up to 32 x 2 MiB = 64 MiB
per sequence on the 27B, inside `margin`; `MADV_NOHUGEPAGE` on the KV would make `kv(hwm)` exact (speed to measure).
State: `free_big` the buffers of free checkpoint slots (`pos < 0`, not being written); at `len == 0` also
`MADV_DONTNEED` the live state (`RqSeq.st_res`: touched since, set by `seq_grow`). `rq_seq_resident`'s `state` is one
copy, `checkpoints` the allocated buffers, `total = kv + (checkpoints + live) x state`. Linux: `MADV_DONTNEED`;
other POSIX systems: `MADV_FREE` where defined; Windows: only the checkpoint buffers go, `hwm` stays (the accounting
stays correct, only less memory comes back).

**Who calls it** (src/engine.c):
- `start_run`, after the truncate / copy: `rq_seq_release(s, E(j))` and the same on the draft. The slot then holds
  at most its job's lease.
- step 1's reclaim and step 4's swap-out: `rq_seq_reset` + `rq_seq_release(s, 0)`.
- after the calibration (step 1).
- `end_run` does **not** release: an idle slot keeps its tokens as a prefix cache until reclaim needs the memory.

Released blocks are faulted in again when the sequence grows into them: prefills pre-fault their tiles
(`kv_populate_one`); decode faults one tile per 16 positions.

## Step 3: a live budget

The static budget does not see the other processes, and the kernel will not refuse us: overcommit is heuristic on
this box (Committed_AS 36.1 GiB against a CommitLimit of 31.3 GiB already), so an overshoot means swap (~7 GiB in
use now) or the OOM killer, not a failed allocation.

At each admission and (step 4) renewal, at most every 100 ms (reading /proc/meminfo takes ~20 us):

```
promised  = committed - sum of res        (what running jobs may still grow into: not in MemAvailable yet)
live      = MemAvailable - margin - promised
growth(j) = lease(j) - res(si after start_run's truncate and release)      (a renewal: kv(E') - kv(E))
admit / renew if   the budget check passes   and   growth(j) <= live
```

MemAvailable already excludes everything we have touched, so only the growth we have promised but not yet touched is
subtracted. `margin`: `--admit-margin GiB`, default 1.5 (other processes' bursts; the sum of `small`; page rounding).

With `--admit bound`, the live budget only stops further admissions: if another process takes memory after a job was
admitted, that job's promised growth may no longer fit, and nothing takes memory back. The engine logs a warning when
`MemAvailable - margin < promised` while jobs run. With `--admit chunk`, a shrinking live budget also denies
renewals, and step 4's swap-out gives memory back: the promise is never more than one chunk per job.

Pages the OS swapped out (an idle slot's cold cache) still count in `res` but are not in RAM. That is conservative for
the static check, **not for the live one** (a known limitation): reclaiming an idle slot whose pages are in the OS swap
is credited as RAM given back (`growth - reclaimed <= live`), but it frees swap, not MemAvailable, so the job can be
admitted (or a renewal granted) beyond what the box really has. `mincore` on the slot's ranges would tell; not done.

As built (engine.c, `mem_admit`, `mem_avail`, `mem_publish`):
- `growth = lease - res(si)` with `res(si)` as the slot holds it now, not after `start_run`: what the truncate and the
  release give back returns to MemAvailable, so it nets out (`promised + lease - res(si) <= MemAvailable - margin`).
- The reclaim counts for both checks: an idle slot's memory, once given back, is in MemAvailable too; the job waits
  only if neither check fits with every eligible idle slot reclaimed.
- Between two readings of /proc/meminfo, what the slots touched (or gave back) since the last one is taken out of (put
  back into) it: `MemAvailable ~ read + res(at the read) - res(now)`, so a pass's new pages are not counted twice (in
  the stale reading and no longer in `promised`).
- `RUNQ_TEST_MEMAVAILABLE=FILE` (tests only) is read at each use, as is (no cache, no adjustment); a file that cannot
  be read means no live budget.
- The warning (`MemAvailable - margin` below `promised` while jobs run) is checked at every engine step, at most once a
  minute in the log; `/health`'s `mem` is a snapshot the engine thread publishes under `mu` at each step (it never
  reads the slots from a connection thread). Without `--admit bound` none of it runs.
- `-v`: step 1's wait line ends with `; live -0.53: MemAvailable 0.00 - margin 0.50 - promised 0.03, it grows 0.08`;
  a job admitted beyond the live budget with nothing running logs a warning, as one over the static budget does.

## Step 4: chunked leases (`--admit chunk`)

A job is admitted on its prompt plus `C` generated tokens (`E = min(P, n_prompt + C)`), and its lease is renewed
chunk by chunk as it generates. `max_tokens` only caps the lease (`E <= P`), so the same rule covers every request.

### Pause the job, do not re-queue a new request

At a lease end the job stays the same `Run` and only waits for more memory. Re-queuing it as a new request (prompt +
output so far) would lose too much:

- **Its in-flight state:** the token sampled but not fed yet (`Run.feed`), the output parser (`Out`: stop strings,
  the reasoning / content split, the tool-call scan), the `json_schema` state (`ConState`), the logprobs, the
  speculation and lookup state (`Run.lk`, the draft's lag), and the response stream itself.
- **The RNG:** a new request draws once per prompt token but the last, so sampled output would change and seeded
  runs would no longer reproduce.
- **A re-prefill on hybrid models:** the server re-prefills a cached prompt's last token for its logits, and a
  Qwen3.5 / 3.8 slot can only go back to a checkpoint: each continuation would re-prefill back to the last one, often
  the whole chunk. On Saluki (prompts of 600-800 tokens prefill in 13-18 s, ~45-55 tok/s), that is ~20 s per
  1,024-token chunk that took ~85 s to generate.

A paused job resumes by feeding its pending token: nothing is recomputed.

### Renewal

In `engine_main`, before the admission loop (renewals and resumes come before new requests), for each running job in
admission order (oldest first), when its next pass could write past its lease (`len + 1 + kk(j) > E(j)`):

1. **Renew:** `E' = min(P, E + C)`. Granted if `committed + kv(E') - kv(E) <= budget` and the growth fits step 3's
   live budget. This is the usual case: no pause, no cost but the arithmetic.
2. **Denied, not the oldest job:** the job is **parked**. `step()` skips it (it gets no rows), it keeps its slot and
   memory, and its renewal is retried every step. A job is parked only at a clean point: between passes, with no
   verify segment in flight (`r->vn == 0`). A job in the middle of a segment finishes it first, within the `kk`
   positions its renewal check already kept free.
3. **Denied, the oldest job:** it never parks. First reclaim idle slots (step 1's reclaim). If that is not enough,
   **swap out** the youngest other job (parked ones first, then running ones at a clean point), and repeat until the
   renewal fits.

New requests are admitted only when no job is parked or swapped out, and only if a reserve stays free after them:
one chunk for every running job (`nrun x kv(C)`, ~41 MiB each on Saluki). Without the reserve, the admissions would
fill the budget and every renewal would swap.

**Liveness:** the oldest job always gets its renewal, because everything else can be moved out and one `-c` job
always fits (the startup check). It therefore finishes, the next one becomes the oldest, and so on. Swaps are
triggered only by the oldest job's renewals, so two young jobs cannot swap each other back and forth (as built, also by
the oldest's swap-in, and with `--admit-yield` a yielded job is swapped out for the one whose turn it is: by design, a
round-robin under memory pressure swaps at each turn).

As built (engine.c, `mem_renewals`, `mem_renew`, `run_reach`, `run_held`; the swap: below):
- **Where:** `engine_main` calls `mem_renewals` under `mu` before the admission loop. `e->run` is in admission order, so
  the first job not done is the oldest; `Run.id` numbers the jobs for the log. A job asks for a renewal when
  `run_reach > E`: `run_reach` = its length + the rest of its feed (a prefill's whole remaining prompt and forced tokens,
  a forced reply whole) + `max(sk, sl)` when this pass may speculate (decoding, one token to feed, no forced reply),
  capped at `P`. The renewal goes by whole chunks to cover it: `E' = min(P, E + C x ceil((reach - E) / C))`.
- **Clean point:** a verify segment is sampled whole within its `step()` (`vn` is 0 again at `verify_done` or
  `end_run`), and `run_reach` covers a prefill and a forced reply whole, so between steps every job is at a clean point
  and a running job with `E >= run_reach` writes nothing past `E`: its decode rows, its guesses (`spec_cands`' cap
  `lim <= E - 1 - len` is a guard only: `run_reach` already kept `kk` free), the draft's rows (its catch-up up to the
  target's length + `seg`, then `t` and the guesses but the last) and the lookup (it only reads the tokens). A prefill
  is never held half-way: a job is checked before its first prefill pass (its forced tokens can pass `n + C`).
- **`step()`:** `run_held` (swapped out, parked, or `run_reach > E`: admitted after this step's renewals, a save pending) gives the
  job no rows: no decode rows, no prompt chunk, no draft catch-up. Parking changes when a job computes, never what: no
  RNG draw, no numerics.
- **The checks:** a renewal whose `kv(E')` equals `kv(E)` (the same KV block) is granted without a check (it costs
  nothing; also when the live budget is negative). Else, as an admission's: `committed + d <= budget` and `d <= live`,
  `d` the growth of the slot's part of `committed`. Not the oldest: denied, it parks (no idle slot is reclaimed for it),
  and so does every younger job whose renewal needs memory in the same step (FIFO). A slot whose job the oldest's renewal
  has just planned to swap out counts as emptied in the younger jobs' checks (`sw_run` empties it before the pass, also if
  the save fails), so a job between the oldest and the swapped-out ones is not parked for memory about to be freed
  (those behind a swapped-out one park anyway: FIFO). The oldest: idle slots reclaimed
  (step 1's, LRU; with `--kv-disk` a save first: the renewal waits for `dk_run`, admissions skip that step), then the
  younger jobs swapped out (`mem_swap_out`, below) until it fits; with nothing left to move it is granted anyway with a
  warning at most once a minute (`the oldest job's lease renewed beyond the live budget (nothing left to swap out)`):
  only the live budget can be short then, one `-c` job fits the static one.
- **Admission:** `mem_admit` refuses while a job is parked or swapped out (`e->npark`, `e->nswap`; also with every job
  swapped out, none in a slot) and adds the reserve `rc = (running + 1) x kv(C)` (this job included; `kv(C)` is one KV
  block for a chunk up to 256) to both checks. In a step with swap I/O the head of the queue counts as waiting for memory
  (`mem_wait_ms`, `/health`'s `waiting_for_memory`).
- **The rest:** the deadline, the client's cancel and the shutdown are checked for parked jobs at each step as for the
  others (`end_run` adds the parked time); `/health`'s `mem.parked` counts them (from the runs, at each publish); the
  timings' `renewals` (granted renewals) and `paused_ms`; `-v`: `[engine] job 7 parked at 247 tokens (45 generated): its
  lease to 267 positions does not fit (+0.027 GiB; budget 8.00, committed 0.25, idle 0.14, live 0.00)` / `behind a
  parked job`, `[engine] job 7 resumed at 247 tokens after 0.37 s (its lease to 267 positions)`, and the admission line
  ends with `; job 7, its lease to 236 of 310 positions`.
- **Test:** the `admit-chunk` group parks with the live budget's hook (MemAvailable set to 0 once the requests run;
  `--kv-mem` large, `--admit-margin 1`: the live budget is then -1 GiB, so the oldest's next block swaps out every younger
  job at once, deterministically): the reserve makes the static budget alone never park a job within its first new block.
- **The lease's check:** at each step `mem_renewals` compares every running job's KV (`rq_seq_resident`, target + draft:
  the high-water mark's blocks) with its lease's and logs `[engine] bug: job N's slot holds ...` past it; the
  `admit-chunk` group fails on that line.

### Swap-out and swap-in

**Out** (the engine thread, between passes, without `mu`, as `dk_run`):
- drop the verify rows' transient checkpoints (`rq_seq_checkpoint_rows_drop`, as `spec_release` does); the regular
  checkpoints go into the file.
- `rq_seq_save` the target's and the draft's sequences to `--kv-disk DIR/swap/<job>.rqseq` (+ `.draft.rqseq`). The
  file holds the KV blocks in use, the tokens, the recurrent state, the checkpoints and the RNG
  (`SeqFileHdr.rng`), so the reload is bit-identical. If the draft's save fails, reset the draft sequence instead: its
  guesses only change the speed.
- `rq_seq_reset` + `rq_seq_release(s, 0)` on the slot. The `Run` stays in memory, detached: `r->si = -1`,
  `r->swapped = 1`, its lease removed from `committed`, its file's resident size remembered.

**In** (before admissions, the oldest swapped-out job first): when its resident size plus one chunk fits (budget and
live), take an idle slot (step 1's reclaim if needed), `rq_seq_load` both files into it, set `r->si` and `slot->run`,
delete the files, and the job continues from its pending token.

**Cost:** writes run at ~1.4 GB/s and reads at up to ~7 GB/s on the dev box (seqfile.c). A Saluki job at 40k
positions is ~2.5 GB (KV 1.65 GB + up to 5 state copies): ~1.8 s out, ~0.4-0.8 s in, against ~85 s for a
1,024-token chunk. Recomputing instead (re-prefilling 40k tokens at ~45-55 tok/s: ~12-15 min) is never worth it, so
there is no recompute path.

**Requirements:** `--admit chunk` needs `--kv-disk`. The swap files are outside `--kv-disk-max` (they belong to
running jobs and cannot be evicted). At the start, the free space must hold the static budget (in the worst case
every job but one is swapped out) + the disk tier's 1 GiB reserve, else the startup refuses `chunk`. **This box had
7.5 GB free on `/` when this was planned** (113 GB on 2026-11-07, after a cleanup): enough for ~3 swapped-out jobs at
40k, not for the whole 15.3 GiB budget, so `chunk` would then have been refused until space was freed or `--kv-mem`
lowered to fit it. `DIR/swap/` is emptied at the start (a
swapped-out job does not survive a restart: its client is gone).

As built (engine.c, "the swap": `mem_swap_out`, `mem_swap_in`, `sw_out`, `sw_in`, `sw_run`, `sw_open`):
- **Planned under `mu`, done without it.** `mem_swap_out` (from the oldest's `mem_renew`, or from `mem_swap_in` for the
  oldest job when it is swapped out itself) only picks the victim (the youngest job in a slot but the oldest, parked ones
  first; between steps every job is at a clean point, `vn == 0`), credits its slot's part of `committed` and marks it
  (`r->swap = 1`, `e->swo`); the renewal is then granted. `mem_renewals` returns 1 (no admission in this step), and the
  engine thread, after `mu` is released and before `step()`, runs `sw_run`: the swap-outs, `dk_run`'s spill / load, then
  the swap-in. So a renewal granted on a swap-out's memory never computes before that memory is given back (the slot is
  emptied even if the save fails), and connection threads (`/health`, SSE keep-alives: `run_stream` waits on the job's
  condition, never on `mu` or the engine) never wait on the disk.
- **Out (`sw_out`):** park time closed; the rows' checkpoints dropped (target and draft, as `spec_release`); the slot's
  resident size, the RNG (`*rq_seq_rng`: it lives in the `RqSeq`, and the file's `SeqFileHdr.rng` restores it too; kept in
  the `Run` as well, since a job not prefilled yet, length 0, has no file but already drew its prompt's RNG steps), the
  lengths and the constraint's mask (`Run.alist`: the slot's `allow` / `list` are the slot's, not the job's) kept; the
  files written (the draft's only if the job uses the draft: a job without it leaves the slot's draft sequence unrelated
  to it); the slot emptied (`rq_seq_reset` + `rq_seq_release(s, 0)`, its checkpoint count back to `--checkpoints`: an idle,
  empty slot). The path is `DIR/<model>-<KV type>/swap/<job id>.rqseq`, not `DIR/swap/`: under the disk tier's lock,
  so two servers on one DIR (other models) never empty each other's swap. `-v`: `[engine] job 7 swapped out: 0.09 GB,
  0.08 s (slot 2: 240 tokens + the draft's, 0.194 GiB given back)`.
- **In (`mem_swap_in`, `sw_in`):** after the renewals, if no I/O is planned in this step, the oldest swapped-out job
  only (FIFO), one per step: on the idle slot holding the least (saved first if the disk tier wants it; then reclaimed),
  checked as an admission on its **lease** (not its resident size: `committed` counts the lease whole once it is back,
  and the lease is >= what its slot held) **+ one chunk** (its next renewal) **+ the oldest job's next renewal** (the
  growth of `kv(min(P, E + C))` over what its slot counts now, if it is in a slot), the last two not for the oldest job,
  other idle slots reclaimed (LRU) if that makes it fit. The last term is the ping-pong guard: without it a job swapped
  back in is swapped out again at the oldest's next KV block. A job back in has at least its next chunk's room; it can
  still be swapped out later if its own renewals (or younger jobs') take the oldest's share, after progress. The oldest job of all (swapped out when the one before it ended) is never left
  out: the idle slots, then the younger jobs swapped out, then in with a warning (`the oldest job swapped in beyond the
  live budget`). A swapped-out job counts as a parked one for the younger ones' renewals (they park behind it: FIFO).
  The load: the checkpoint slots the job had (`spec_start`'s count: every saved checkpoint comes back), `rq_seq_load`,
  the RNG, the files deleted; the job's pending checkpoints asked for again (`ckw` past the length; the draft's:
  `draft_ck_pending`), the mask back into the new slot. `-v`: `[engine] job 7 swapped in -> slot 5: 0.08 s (242 tokens +
  the draft's; out 0.23 s)`.
- **Detached runs** (`r->si = -1`): `run_held` gives them no rows (so `step()`, the speculation's candidates and the
  draft's catch-up skip them), `end_run` deletes their files instead of touching a slot (the cancel, the deadline and
  the shutdown are checked for them in `step()` as for the others), the lease check, `mem_admit`'s and
  `mem_swap_in`'s sums and `slot_mem` see only slots, `admit()`'s pending-prefix scan skips them, `reap` frees `alist`.
  `/health`: `mem.swapped_out` (planned ones included); timings: `swaps`, `swap_ms` (the time out of memory, from the
  swap-out's start to the swap-in's end).
- **Failures:** a target's save that fails (disk full, I/O error) or a mask that cannot be kept: that job ends with a
  500 (`the request could not be swapped out to disk (--kv-disk)`), its files deleted, its slot emptied all the same; a
  load that fails: the same (`... swapped back in ...`). A draft's save / load that fails resets the draft (speed only).
  No other job is touched.
- **Disk room:** `sw_open` refuses `chunk` unless free space + the disk tier's files >= `--kv-mem` + the disk tier's
  1 GiB reserve (those files can be evicted), and in chunk mode the disk tier keeps `--kv-mem` - the swap files' bytes
  free beyond that reserve (`dk_fit`), so its saves do not eat the swap's room. Other processes can still fill the
  disk: then a swap-out fails as above.
- **Prefill:** a job can be swapped out between two passes of its prefill (the file holds the positions prefilled so
  far; `-v` adds `prefilling: N of M prompt tokens`); it continues the prefill after the swap-in (the chunking does not
  change the logits). A job not prefilled at all has no file (its RNG is kept in the `Run`).
- `RUNQ_TEST_SWAP_DELAY=s` (tests only): a sleep in each swap-out, so that a swap lasts (the SSE check).
  `RUNQ_TEST_SWAP_FAIL=FILE` (tests only): read at each swap; `draft-save` / `draft-load` in it make the draft's save /
  load fail (logged: `[engine] job N: the draft's sequence could not be swapped out ...` / `... back in ...`).

### The rest of the job's life

- **Streaming:** the SSE keep-alive (`: keep-alive` every 15 s, `RUNQ_SSE_KEEPALIVE`) already covers pauses and
  swaps (the connection thread waits on its job, 0.1 s at a time, never on the engine); non-streaming clients just wait
  longer.
- **`--timeout`:** the deadline counts the paused and swapped-out time, as it counts the queue's today.
- **Cancel** (the client left) while parked or swapped out: finish the job, delete its files.
- **Prefill:** a long prompt is covered by its first lease (`E >= n_prompt`); a prefill is never parked mid-way in v1
  (as built it can be swapped out between two of its passes: the chunking does not change the logits).
- **Speculation:** a pass's rows stay within `E` (renewal before the pass, `kk` included in the check), so the cap
  on guesses (`max_new`) and the lease agree.

### Optional: yielding the compute seat (`--admit-yield N`, default off)

At a lease end, if requests wait for a seat (`nrun == --parallel`) and this job has generated at least N chunks
since it started or resumed, park it (or swap it out, if the waiting job needs its memory) and admit the waiting
one: a round-robin of long generations. It makes long jobs slower and short ones faster, and a parked job still holds
its memory. Off by default; worth it only with a mix of interactive and batch clients (IDEAS.md, request priorities).

As built (engine.c, `mem_yield`, `mem_resume`, in `mem_renewals`; `--admit-yield` needs `--admit chunk`):
- **Seats:** a yielded job (`r->yld`) holds no seat: the seats taken are `nrun - nyld` (`e->run` holds up to
  `parallel + --queue` jobs). It keeps its slot and memory, gets no rows (`run_held`), asks for no renewal and holds
  back no younger job's renewal (it waits for a seat, not for memory).
- **When:** at a lease end (`run_reach > E`), with every seat taken and someone waiting for one (a yielded job, or the
  queue while admissions are not held by a job parked or swapped out for memory), a job that has generated N chunks
  (N x `--admit-chunk` tokens) since it started or last resumed (from a yield, a park or a swap) yields. A job whose
  renewal is denied while a yielded job waits yields instead of parking (parked, it would keep the seat that job
  needs: with every seat parked, a deadlock).
- **Who gets a seat:** the one waiting longest, FIFO by the time it began to wait: the yielded jobs (their yield time)
  against the queue's head (its submission). A yielded job in its slot resumes at once; a swapped-out one is swapped in
  (`mem_swap_in`, one per step); the queue's jobs that came after a waiting yielded job's yield wait behind it.
- **Memory:** if the job whose turn it is does not fit (or no slot is idle: yielded jobs hold them), the youngest
  yielded job in a slot is swapped out for it, one per step, until it does (for the queue's head: when it is refused
  and no job is parked or swapped out for memory). A yielded job comes back in as a swapped-out one does (its lease +
  a chunk + the oldest's next renewal).
- **Liveness:** the oldest job yields its seat like the others (otherwise a round-robin with it is impossible) but keeps
  its precedence for memory: it is never parked, its renewals swap the others out (yielded ones first, with the parked
  ones), and its swap-in, when its turn comes, goes in whatever it takes; so does any job's when no other job with a
  seat is in a slot (else, under a live budget below one job, nothing could run). Every turn ends: after N chunks, or at
  a denied renewal (a yield when a yielded job waits; the oldest is never denied), so the oldest gets a seat back after
  at most one turn of each job ahead of it in the FIFO, and with a seat it always progresses: it finishes, the next one
  becomes the oldest. FIFO holds among the yielded jobs (their yield times) and, as before, among the jobs parked or
  swapped out for memory (admission order).
- **Output:** a yield changes when a job computes, never what (the same clean point as a park: no RNG draw).
- `/health`'s `mem.yielded` (jobs without a seat; `parked` counts the memory-parked ones only, `swapped_out` every
  swapped-out job), `running` counts them all; the timings' `yields`, and `paused_ms` includes the time without a seat;
  `-v`: `[engine] job 3 yielded its seat at 73 tokens (49 generated, 49 since it resumed): a request waits in the queue`,
  `[engine] job 3 resumed at 73 tokens after 0.22 s without a seat`.
- **Test:** the `admit-chunk` group, `--parallel 1 --admit-yield 2`: two long requests take turns (their yields
  alternate in the log), both with `yields`, `paused_ms` and `renewals`, each == alone bitwise (greedy; seeded with
  speculation); the same with MemAvailable 0 once both run: the yielded one is swapped out for the other (`swaps`).

## Options, output

| option | default | |
|---|---|---|
| `--admit worst\|bound\|chunk` | `worst` | `worst`: today's pool. `bound`: steps 1-3. `chunk`: step 4 (needs `--kv-disk`). The default changes only after the Saluki measurements |
| `--admit-chunk N` | 1,024 | step 4's chunk in generated tokens |
| `--admit-margin GiB` | 1.5 | step 3's margin |
| `--admit-yield N` | 0 (off) | `chunk`: yield the compute seat after N chunks when requests wait for one (round-robin) |
| `--kv-mem GiB` | as today | `bound`, `chunk`: the static cap on `committed` |

- The startup line with `bound` / `chunk`: the mode, the static budget, the margin, `per position 41.3 KB, state
  168.9 MiB`, the largest job (`-c`, speculating) that fits; with `chunk` the chunk and the swap directory's free space
  (and the yield's N). With `--draft mtp` the slot line names the view's sequence (`the MTP view's sequence X MiB
  (blk.N, one attention layer)`) and the terms are target + view (Qwen3.8-27B q8: 37.0 KB, 149.6 MiB).
- The options take `--draft mtp` as any draft (no option of its own).
- `/health`: `"mem": {"budget", "live", "committed", "resident", "waiting_for_memory", "parked", "swapped_out"}` in
  GiB / jobs (`bound`: without `parked`, `swapped_out`; `live` is `null` without MemAvailable), + `"yielded"` with
  `--admit-yield`.
- The timings: `mem_wait_ms` (the part of `queue_ms` spent waiting for memory), `lease_gib` (at the end), and with
  `chunk`: `renewals`, `paused_ms`, `swaps`, `swap_ms`, `yields`.
- `--verbose`: `[engine] job waits for memory: needs 4.76 GiB, 2.10 free (budget 15.30, committed 13.20)`,
  `[engine] slot 3 reclaimed: 61,440 tokens, 3.1 GiB (saved to disk)`, `[engine] job 17 parked at 12,288 tokens`,
  `[engine] job 17 swapped out: 2.48 GB, 1.79 s` / `swapped in -> slot 2: 0.52 s`.

## Tests

- **api_test (C), `rq_seq_release`:** fill a sequence to n, release to `keep` (block-aligned, mid-block, 0): RssAnon
  drops by `kv(n) - kv(keep)` within a page per range; `rq_seq_resident` agrees; the next tokens' logits are
  bitwise those of a sequence that was never released (fresh, and continued from the same prefix). Hybrid
  (synthetic Qwen3.5): free checkpoint buffers freed, kept checkpoints still restore (`rq_seq_truncate` to them),
  `len 0` frees the live state. The generic build too.
- **server_test, new group `admit`** (Qwen3-0.6B and the Qwen3.5-0.8B file; `--admit bound` with a `--kv-mem` where
  one `-c` job fits but three short ones do too):
  - three short requests (`max_tokens` 16) at once all run together (`queue_ms` ~0, `/health` running 3); their
    outputs and logprobs equal the same requests alone.
  - a request without `max_tokens` sent while they run waits (`mem_wait_ms` > 0) and then completes, identical to
    alone.
  - strict FIFO: a short request behind the waiting one waits too.
  - reclaim: idle slots holding long prompts are reclaimed when a job needs the room; with `--kv-disk` they are saved
    first and the conversation's next request loads them back (`cached` > 0).
  - nothing running: a job that needs everything is admitted after reclaiming every idle slot.
  - RssAnon of the server never exceeds weights + runtime + `committed` + a page per range.
- **Live budget:** a test hook, `RUNQ_TEST_MEMAVAILABLE=FILE` (bytes, read at each admission instead of
  /proc/meminfo): lowering it makes new jobs wait (and with `chunk`, renewals swap), raising it lets them in.
- **server_test, group `admit-chunk`** (`--admit chunk --admit-chunk 16 --kv-disk`, a `--kv-mem` where the replies'
  growth does not fit together):
  - parked and swapped-out mid-reply, every reply equals the same request alone, bitwise with logprobs: greedy and
    seeded; thinking (the `</think>` inside a chunk and on its boundary); tool calls and stop strings spanning a
    boundary; `json_schema`; speculation on (draft and lookup) and off; Qwen3 and Qwen3.5.
  - liveness: three requests without `max_tokens` whose growth exceeds the budget all finish, the oldest first;
    `swaps` > 0.
  - no new admission while a job is parked or swapped out.
  - cancel while swapped out: the job ends, its files are deleted. A restart empties `DIR/swap/`.
  - SSE: keep-alive comments arrive during a swap.
  - as built, also: a swap-out that fails (a 500 for that request only); a swap mid-prefill (`-c 4096 --chunk 8`, a
    ~2,400-token prompt swapped out by the oldest's block and continued after the swap-in); the draft's save, then its
    load, made to fail (`RUNQ_TEST_SWAP_FAIL`): the draft reset, the outputs unchanged; `--admit-yield` (above).
- **With the MTP view** (TODO S9c.3 / S9c.4; tests/check_spec.py, the spec group, on the ISTA-shaped synthetic file):
  `spec_server/mtp_admit/*` (`server_test.py --admit --draft mtp`: the budgets from the startup line as above, so the
  view's terms enter by themselves; `admit/mtp_terms`: the leases' KV per position == the target's + the view's, the
  state == the target's; no `[engine] bug:` line under `RUNQ_TEST_MTP_CHECK`) and `spec_server/mtp_admit_chunk/*` (the
  view swapped with every job, its kept state in the version 2 file).
- `--admit worst` (the default): every existing group unchanged.

## Measurements (the 16-vCPU box)

TODO: not run yet (they need the 27B model; TODO.md, Server). The default stays `worst` until they are.

1. Saluki q8-dot, `--admit bound --parallel 3 --slots 5`: the benchmark of the 1-3 request table (prompts 600-800,
   1,024 tokens). Expected: 3 at once at q8 like q4's 3 slots (~19 tok/s total), with ~5.7 GiB committed instead of
   16.8. Peak RSS against the estimate.
2. `--parallel 4`: whether a fourth request still pays (compute, not memory, is expected to be the limit).
3. An agent session (`scripts/replay_sessions.py`, 30-90k) with a stream of short requests beside it: waits, reclaims,
   the session's re-prefills (do reclaims cost it its cache?), `mem_wait_ms`.
4. The same with `--kv-disk` (launch-server passes `--kv-disk ./kv-cache --kv-disk-max 10` since 2026-11-07).
5. `--admit chunk`: the same mix with requests without `max_tokens` (an eval harness's): admitted at once instead of
   waiting, `renewals` / `swaps` / `paused_ms` per request, swap time against generation time, with the box's free
   disk space as the limit.

## Implementation order

Step 2 first (it is what makes more slots than the worst case safe), then steps 1 and 3, measured, then step 4:

1. `RqSeq.hwm`, `seq_grow`, `rq_seq_resident`, `rq_seq_kv_bytes`, `rq_seq_release` + api_test.
2. The engine's accounting (`lease` with `E`, `res`, `committed`: recomputed at each admission, step 1's "As
   built"), `--admit bound`, the reclaim and the wait in `admit()`, the startup changes, the calibration's release +
   the `admit` group.
3. The live budget, `--admit-margin`, the test hook, `/health` and the timings.
4. docs/server.md (options, Limitations: "a fixed pool" becomes conditional), measurements 1-4, then the decision on
   the default.
5. Step 4: the renewal and parking in `engine_main` / `step()` (park only, no swap yet: the `admit-chunk` group's
   bit-identity checks with a budget the replies fit in), then the swap-out / swap-in, `DIR/swap/`, the liveness and
   cancel checks, `--admit-yield` last; measurement 5.

Done: 1-3, 5 and item 4's docs (2026-11-07); left: measurements 1-5 and the default (TODO.md).

## Open questions

- **Head-of-line blocking:** a job without `max_tokens` (lease = `-c` with `bound`) at the head of the queue holds
  back the short ones behind it. Strict FIFO first; step 4 removes most of it (its first lease is one chunk); then
  maybe let a fitting job pass a waiting head that has waited less than N seconds (IDEAS.md, request priorities).
- **Partial reclaim:** release an idle slot only past its system-block checkpoint (`ck_plan_job`'s), keeping the
  part every agent request shares, before reclaiming it whole.
- **Fewer state copies for short jobs:** 11 x 168.9 MiB per speculating job on Saluki, reserved whole in every lease.
  One-shot requests may not need 4 checkpoints, and `kk + 1` transient copies are only touched as guesses are
  verified. A per-job count would make short jobs ~2x cheaper; chunking does not change this part.
- **The chunk size:** 1,024 tokens is 41 MiB on Saluki, so a larger chunk costs little and means fewer renewals; a
  smaller one parks later. Measure before tuning.
- **Swapping to a full disk:** with 7.5 GB free, a swap-out can fail. Then the oldest job's renewal cannot be
  guaranteed: v1 refuses `chunk` at startup unless the free space holds the budget; a lower bound (swap only what fits,
  park the rest) needs its own liveness argument.
