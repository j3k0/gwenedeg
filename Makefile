# cwayn: Qwen3-0.6B inference in C (engine: src/runq.c, derived from qwen3.c)

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

SRC = src/runq.c
HDR = src/unicode_tables.h src/unicode_nfc.h src/kernels_avx512.h src/kernels_neon.h src/kquants.h src/threads.h
Q8  = models/Qwen3-0.6B-q8.bin

.PHONY: all debug generic clean test bench bench-prefill models kq-neon-test

all: runq

runq: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LDLIBS) -o $@

runq-debug: $(SRC) $(HDR)
	$(CC) $(DBGFLAGS) $(CPPFLAGS) $< $(LDLIBS) -o $@

# Same flags, but the generic C kernels instead of the AVX-512 VNNI ones (kernels_avx512.h);
# `make test` checks that both give bit-identical logits. (On a CPU without AVX-512 VNNI,
# plain `make` already builds the generic kernels: the selection is at compile time.)
runq-generic: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -DRUNQ_GENERIC $< $(LDLIBS) -o $@

debug: runq-debug

# NEON kernels (kernels_neon.h: K-quants, Q8_0) vs the generic ones, on random data: bit-identical
kq-neon-test: tests/kernels/kq_neon_test.c src/kernels_neon.h src/kquants.h
	$(CC) $(CFLAGS) $< -o tests/kernels/kq_neon_test && tests/kernels/kq_neon_test

generic: runq-generic

clean:
	rm -rf runq runq-debug runq-generic runq-debug.dSYM tests/kernels/kq_neon_test

# (run_tests.sh builds runq and runq-generic itself, in parallel)
test:
	@if [ -f tests/run_tests.sh ]; then bash tests/run_tests.sh; else echo "no tests yet"; fi

bench: runq
	scripts/bench_qwen3c.sh

bench-prefill: runq
	scripts/bench_prefill.sh

# Downloads ~2 GB and runs the Python exporter; only if the Q8_0 model is missing.
models:
	@if [ -f $(Q8) ]; then echo "$(Q8) present, nothing to do"; \
	else scripts/download_models.py && scripts/export_q8.sh; fi
