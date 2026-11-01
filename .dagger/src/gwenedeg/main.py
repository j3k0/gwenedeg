"""Gwenedeg CI: build runq and run the regression suite in containers, cached by content.

    dagger check                  # every check (build, test), in parallel; cached results return at once
    dagger check -l               # list them
    dagger check test             # one check
    dagger call test --only=server,api        # a subset of check.py's groups (cached per subset)
    dagger call build export --path=./build-ci  # the binaries on the host

Each step's cache key is its inputs: the files below (by content, not mtime), the toolchain image, the
command and the host CPU. Change a comment in README.md: nothing reruns. Change src/forward.c: the build
reruns, then the tests. Change only tests/server_test.py: the build is a cache hit, the tests rerun.
"""

import hashlib
from typing import Annotated

import dagger
from dagger import DefaultPath, Doc, Ignore, check, dag, function, object_type

# Debian 13, as the dev box (gcc 14.2, Python 3.13); pinned so a base image update cannot bust the cache
BASE = "debian:trixie-slim@sha256:f6b08b09a05725f9431502d3b5714f5c4576e82c1a84fa1debe0c24a7e0c3ec4"

# What the build and the fast suite read (tests/run_tests.sh, tests/check*.py): an allow-list, so docs,
# benchmarks, logs, binaries, .venv, *.key and the 53 GB of models/ never reach the engine or the cache key
SOURCE_IGNORE = [
    "*",
    "!Makefile",
    "!src",
    "!tests",
    "**/__pycache__",
    "tests/README.md",  # docs: editing them reruns nothing
    "tests/kernels/kq_neon_test",
    "tests/kernels/iq_neon_test",
]

# The model files of make test's groups (README.md, tests/README.md): ~3.5 GB of the 53 GB in models/.
# A missing one is skipped by run_tests.sh with a note, as on the host
MODEL_FILES = [
    "Qwen3-0.6B-q8.bin",
    "Qwen3-0.6B-q8.bin.*",  # qwen3.c's .tokenizer / .template* files next to it
    "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
    "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
    "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf",
    "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
    "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q3_K_M.gguf",
]
MODELS_IGNORE = ["*"] + [f"!{f}" for f in MODEL_FILES]

BINARIES = ["runq", "runq-generic", "api-test", "tools-test", "build/bf16dp_test", "build/bf16dp_test-emu"]


def _host_cpu() -> str:
    """The CPU the engine runs on (model + flags). The Makefile builds with -march=native, so a cached
    binary is only valid on the same CPU: this goes into the build's cache key (matters with a shared
    cache, e.g. Dagger Cloud, or several runners)."""
    try:
        with open("/proc/cpuinfo") as f:
            lines = [ln for ln in f if ln.startswith(("model name", "flags", "Features", "CPU part"))]
        info = "".join(dict.fromkeys(lines))  # one copy per distinct line
    except OSError:
        info = "unknown"
    return hashlib.sha256(info.encode()).hexdigest()[:16]


@object_type
class Gwenedeg:
    source: Annotated[
        dagger.Directory,
        DefaultPath("/"),
        Ignore(SOURCE_IGNORE),
        Doc("the repository: Makefile, src/, tests/ only"),
    ]
    models: Annotated[
        dagger.Directory,
        DefaultPath("/models"),
        Ignore(MODELS_IGNORE),
        Doc("models/: the files of the fast suite only"),
    ]

    @function
    def toolchain(self) -> dagger.Container:
        """gcc, make, Python 3 + numpy: what make and tests/run_tests.sh need."""
        return (
            dag.container()
            .from_(BASE)
            .with_exec(
                [
                    "sh",
                    "-c",
                    "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
                    "gcc libc6-dev make python3 python3-numpy >/dev/null && rm -rf /var/lib/apt/lists/*",
                ]
            )
            .with_env_variable("RUNQ_HOST_CPU", _host_cpu())
        )

    @function
    def builder(self) -> dagger.Container:
        """The toolchain with the sources built: runq, runq-generic, the API / tools / bf16 tests."""
        return (
            self.toolchain()
            .with_directory("/src", self.source)
            .with_workdir("/src")
            .with_exec(["make", "-s", "-j16", *BINARIES])
        )

    @function
    @check
    async def build(self) -> dagger.Directory:
        """Compile everything make test needs (fails fast on a compile error, before the suite)."""
        b = self.builder()
        out = dag.directory().with_file("runq", b.file("runq")).with_file("runq-generic", b.file("runq-generic"))
        return out.with_directory("build", b.directory("build"), include=["api_test*", "tools_test", "bf16dp_test*"])

    @function
    @check
    async def test(
        self,
        only: Annotated[str, Doc("check.py groups, comma-separated (default: all of make test's)")] = "",
        threads: Annotated[int, Doc("OMP_NUM_THREADS, as run_tests.sh's default")] = 8,
    ) -> str:
        """The fast regression suite (make test / tests/run_tests.sh); returns its report."""
        args = ["bash", "tests/run_tests.sh"] + (["--only", only] if only else [])
        return await (
            self.builder()
            .with_mounted_directory("/src/models", self.models)
            # scratch on tmpfs, not in the cached snapshot: the logits files and the synth group's ~1.5 GB of
            # synthetic models (tests/make_synth*.py, rebuilt in ~seconds); only the report is kept
            .with_mounted_temp("/tmp")
            .with_mounted_temp("/src/build/synth")
            .with_mounted_temp("/src/build/synth-q35")
            .with_env_variable("OMP_NUM_THREADS", str(threads))
            .with_exec(args)
            .stdout()
        )
