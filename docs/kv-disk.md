# The KV cache on disk (`runq serve --kv-disk DIR`)

For operators of `runq serve`. This is the section of [server.md](server.md) that described the disk tier, moved here
unchanged (2026-11-21) except for its first sentence and the pointer to the command it refers to; the flags (`--kv-disk`, `--kv-disk-max`, `--kv-disk-free`,
`--kv-disk-min`) stay in [server.md](server.md)'s flag table. The disk is also where `--admit chunk` swaps requests out
([memory-admission.md](memory-admission.md)). Measurements are dated 2026-10-31 on the dev box's virtual SSD.

The KV cache's **disk tier** (`--kv-disk DIR`) is a second tier below the slots, for sequences the pool has no room to keep
(e.g. one 96k slot: a compaction's summary request takes it, the next request then prefilled the system block again)
and across restarts. A slot is saved (`rq_seq_save`: the KV blocks in use, the tokens, for Qwen3.5 / 3.8 the
recurrent state and its checkpoints; with `--draft`, the draft's sequence beside it) when a request is about to
discard at least `--kv-disk-min` of its tokens (an eviction, a truncation, a copy over it), once per change of its
contents, and every slot at shutdown (SIGTERM). The saved tokens are indexed in memory; a request whose prompt
shares a longer usable prefix with a saved sequence than with any slot (by >= `--prefix-copy` tokens and 1/8 of the
length read; Qwen3.5 / 3.8: the saved length or a checkpoint) gets that prefix loaded into the least recently used
idle slot (that slot saved first if it needs to be; only the blocks up to the prefix are read), then is admitted
as for a cached prefix: its prefill starts there. Logits are bit-identical to a fresh prefill (the same bits come
back; tested, tests/server_test.py `--disk`). A save whose tokens extend an older one's (the same conversation,
further on) replaces it: one file per conversation (Qwen3.5 / 3.8: the lengths the slot gave up checkpoints for, e.g.
the previous reply's end, are given up on disk too). The files are in `DIR/<model fingerprint>-<KV type>/` (`<id>.rqseq`, `<id>.draft.rqseq`;
the fingerprint: the model file's size, first MiB and last 64 KiB), one server per directory (a lock file; a
second one is refused); at the start, saves cut short (`.tmp`) and files that do not check out (corrupt, another
model) are removed, and the rest are indexed with their mtime as their last use. Written as `.tmp`, synced,
renamed; O_DIRECT where the file system takes it (the page cache is not filled with them; tmpfs: buffered), one
thread per layer and K / V. The MTP drafter's file (`--draft mtp`, TODO S9c; the view at rest keeps the target's
last hidden state) is format version 2: that state is in it too, under the file's hash (+20 KB per conversation on
Qwen3.8-27B: dim 5,120 floats in one 4 KiB-aligned section), so a loaded or swapped-in view goes on exactly. A loaded
view goes to the prompt's common prefix with the saved tokens (TODO S9c.4), not to the target's loaded length: a
hybrid target restarts at a checkpoint below it, and the view's rows above wait for the prefill (their h was the
target's); the whole file, its kept state too. A draft file that does not check out against the server's drafter
(another draft model's, e.g. after switching to `--draft mtp`; another KV type; other tokens) is removed at the
start, logged (`runq serve: --kv-disk: DIR/N.draft.rqseq: not the draft's (...), removed`), its target file kept: such
a conversation's view then follows the loaded prefix as a gap (zeros for h, counted in `mtp_approx`; outputs exact). Every
other file is version 1, byte for byte as before (tested against 915894a's: `seqfile/v1_same`, and `server_test.py
--disk-compat`: one directory served by an older build, this one and the older one again, nothing removed); a
build before version 2 takes a version 2 file for one that does not check out and removes that draft file only (its
target file stays and loads). A file holds the blocks of 256 positions in use: 115 KB per position for the 0.6B
(f16), 64 KiB for Qwen3.8-27B (its 16 attention layers; + 150 MiB per recurrent state, ~6.6 GiB for a 98k session
with 3 checkpoints). The I/O runs between passes, without the queue's lock (queueing and `/health` never wait for
it), but the running requests do: on the dev box's disk (a virtual SSD; `dd` 1.4 GB/s written, 7.3 GB/s read with
O_DIRECT), against a prefill of the same context (16 threads, f16, 2026-10-31; the next decode step's logits after
the load == the saved sequence's):

| model, context | prefill | file | save | load |
|---|---|---|---|---|
| Qwen3-0.6B Q8_0, 16k | 31.1 s (527 tok/s) | 1.91 GB | 0.63 s | 0.26 s (117x faster) |
| Qwen3.5-0.8B Q8_0, 32k | 34.9 s (940 tok/s) | 0.43 GB | 0.16 s | 0.08 s (427x) |
| Qwen3.5-4B Q8_0, 16k | 44.0 s (372 tok/s) | 0.60 GB | 0.21 s | 0.13 s (334x) |

**Qwen3.8-27B** (Q4_K_M, `-c 32768 -k bf16-dot --slots 1 --checkpoints 3`, 16 threads, no draft; 2026-10-31): a
27k-token conversation (a 2,838-token system block + a source file) prefilled in 408-418 s; a summary-like request
then took the slot (the conversation saved first: 2.41 GB, 0.74-0.80 s); a compaction-like request on the same
system prompt got the system block from the file (2,838 tokens, 0.12 s; 3.5 s to its reply instead of ~45 s); the
conversation's next turn loaded 27,017-27,065 tokens in 0.58-0.61 s (its reply 4.5-4.9 s after the request instead of
~7 min); after a restart (the shutdown's save 1.9-2.0 s) the same turn loaded in 0.42 s (3.9 s to the reply). Its
logprobs == the same request prefilled from nothing (414.5 s), via the disk and after the restart. A file holds 64
KiB per position + 150 MiB per recurrent state (1 + 3 checkpoints): ~6.6 GiB for a 98k session, ~2.2 s to save and
~1.6 s to load at these rates. Use: add `--kv-disk DIR --kv-disk-max GiB` to the command in [server.md](server.md#qwen38-27b-with-pi) (DIR on the SSD; the
budget is a hard cap, the room for a save made before it is written).
