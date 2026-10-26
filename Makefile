# cwayn: Qwen3-0.6B inference in C (engine: src/*.c, derived from qwen3.c; modules: README.md)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
# macOS (Apple clang): no OpenMP needed (threads.h's pthread pool; OpenMP is only used with
# -DRUNQ_NO_POOL), -O3 -ffast-math = gcc's -Ofast (deprecated in clang). -ffp-contract=on after it:
# -ffast-math's fp-contract=fast is also a backend option that fuses a * b + c into an FMA across
# statements and functions whatever the source says, so the exact functions (RQ_FP_EXACT, the
# arithmetic definitions of kquants.h / the attention) would round differently in each build
CC      = clang
CFLAGS  ?= -O3 -ffast-math -ffp-contract=on -Wno-overriding-option -mcpu=native
DBGFLAGS ?= -g -O0
else
CC      = gcc
# Same flags as upstream qwen3.c's `openmp` target.
CFLAGS  ?= -Ofast -fopenmp -march=native
DBGFLAGS ?= -g -O0 -fopenmp
endif
CPPFLAGS += -D_FILE_OFFSET_BITS=64
LDLIBS  += -lm -lpthread

# the library modules (runq.h) and the CLI; each module is its own translation unit, all numeric code
# of the forward passes is in forward.c (README.md, Architecture)
LIB = model gguf forward tokenizer sampler chat threads
# runq serve (an OpenAI-compatible HTTP server; serve.h): linked into the runq binary, not part of the library
SRV = json constrain http engine serve
HDR = $(wildcard src/*.h)
Q8  = models/Qwen3-0.6B-q8.bin
B   = build

.PHONY: all debug generic clean test test-models bench bench-prefill models kq-neon-test api-test bench-batch FORCE

all: runq

# objects per build flavour: $(B)/native (CFLAGS), $(B)/generic (+ -DRUNQ_GENERIC), $(B)/debug
# (DBGFLAGS); a .flags stamp per flavour rebuilds its objects when the flags change
$(B)/native/.flags: FORCE
	@mkdir -p $(@D); f='$(CC) $(CFLAGS) $(CPPFLAGS)'; echo "$$f" | cmp -s - $@ || echo "$$f" > $@
$(B)/generic/.flags: FORCE
	@mkdir -p $(@D); f='$(CC) $(CFLAGS) $(CPPFLAGS) -DRUNQ_GENERIC'; echo "$$f" | cmp -s - $@ || echo "$$f" > $@
$(B)/debug/.flags: FORCE
	@mkdir -p $(@D); f='$(CC) $(DBGFLAGS) $(CPPFLAGS)'; echo "$$f" | cmp -s - $@ || echo "$$f" > $@

$(B)/native/%.o: src/%.c $(HDR) $(B)/native/.flags
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
$(B)/generic/%.o: src/%.c $(HDR) $(B)/generic/.flags
	$(CC) $(CFLAGS) $(CPPFLAGS) -DRUNQ_GENERIC -c $< -o $@
$(B)/debug/%.o: src/%.c $(HDR) $(B)/debug/.flags
	$(CC) $(DBGFLAGS) $(CPPFLAGS) -c $< -o $@

runq: $(LIB:%=$(B)/native/%.o) $(SRV:%=$(B)/native/%.o) $(B)/native/cli.o
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

runq-debug: $(LIB:%=$(B)/debug/%.o) $(SRV:%=$(B)/debug/%.o) $(B)/debug/cli.o
	$(CC) $(DBGFLAGS) $^ $(LDLIBS) -o $@

# Same flags, but the generic C kernels instead of the AVX-512 VNNI ones (kernels_avx512.h);
# `make test` checks that both give bit-identical logits. (On a CPU without AVX-512 VNNI,
# plain `make` already builds the generic kernels: the selection is at compile time.)
runq-generic: $(LIB:%=$(B)/generic/%.o) $(SRV:%=$(B)/generic/%.o) $(B)/generic/cli.o
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

debug: runq-debug

# the library API test (tests/api_test.c; make test runs it on each model)
$(B)/api_test: tests/api_test.c src/runq.h $(LIB:%=$(B)/native/%.o)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB:%=$(B)/native/%.o) $(LDLIBS) -o $@

# the same on the generic kernels (make test compares the batched passes' digests with the native build's)
$(B)/api_test-generic: tests/api_test.c src/runq.h $(LIB:%=$(B)/generic/%.o)
	$(CC) $(CFLAGS) $(CPPFLAGS) -DRUNQ_GENERIC $< $(LIB:%=$(B)/generic/%.o) $(LDLIBS) -o $@

api-test: $(B)/api_test $(B)/api_test-generic

# the batched decode benchmark (tests/bench_batch.c; scripts/bench_batch.sh)
$(B)/bench_batch: tests/bench_batch.c src/runq.h $(LIB:%=$(B)/native/%.o)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB:%=$(B)/native/%.o) $(LDLIBS) -o $@

bench-batch: $(B)/bench_batch

# NEON kernels (kernels_neon.h: K-quants, Q8_0) vs the generic ones, on random data: bit-identical
kq-neon-test: tests/kernels/kq_neon_test.c src/kernels_neon.h src/kquants.h
	$(CC) $(CFLAGS) $< -o tests/kernels/kq_neon_test && tests/kernels/kq_neon_test

generic: runq-generic

clean:
	rm -rf runq runq-debug runq-generic runq-debug.dSYM tests/kernels/kq_neon_test $(B)

# (run_tests.sh builds runq and runq-generic itself, in parallel)
test:
	@if [ -f tests/run_tests.sh ]; then bash tests/run_tests.sh; else echo "no tests yet"; fi

# the bigger Qwen3 models (tests/check_models.py, group models): every downloaded family file, minutes
test-models:
	@bash tests/run_tests.sh --models

bench: runq
	scripts/bench_qwen3c.sh

bench-prefill: runq
	scripts/bench_prefill.sh

# Downloads ~2 GB and runs the Python exporter; only if the Q8_0 model is missing.
models:
	@if [ -f $(Q8) ]; then echo "$(Q8) present, nothing to do"; \
	else scripts/download_models.py && scripts/export_q8.sh; fi
