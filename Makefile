# cwayn: Qwen3-0.6B inference in C (engine: src/runq.c, derived from qwen3.c)

CC      = gcc
# Same flags as upstream qwen3.c's `openmp` target.
CFLAGS  ?= -Ofast -fopenmp -march=native
DBGFLAGS ?= -g -O0 -fopenmp
CPPFLAGS += -D_FILE_OFFSET_BITS=64
LDLIBS  += -lm

SRC = src/runq.c
Q8  = models/Qwen3-0.6B-q8.bin

.PHONY: all debug clean test bench models

all: runq

runq: $(SRC)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LDLIBS) -o $@

runq-debug: $(SRC)
	$(CC) $(DBGFLAGS) $(CPPFLAGS) $< $(LDLIBS) -o $@

debug: runq-debug

clean:
	rm -f runq runq-debug

test: runq
	@if [ -f tests/run_tests.sh ]; then bash tests/run_tests.sh; else echo "no tests yet"; fi

bench: runq
	scripts/bench_qwen3c.sh

# Downloads ~2 GB and runs the Python exporter; only if the Q8_0 model is missing.
models:
	@if [ -f $(Q8) ]; then echo "$(Q8) present, nothing to do"; \
	else scripts/download_models.py && scripts/export_q8.sh; fi
