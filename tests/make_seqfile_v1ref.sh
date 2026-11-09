#!/usr/bin/env bash
# tests/make_seqfile_v1ref.sh DIR: the api_test of 915894a (seqfile.c's version 1 only: before TODO S9c.2) with this
# tree's --seqfile-out / --seqfile-load modes (tests/api_test.c's "seqfile-out" block, inserted before its main), built
# with the Makefile's flags as DIR/build/api_test (tests/run_tests.sh: DIR build/seqfile-v1ref; nothing done when DIR's
# stamp names the same commit, block and script). For tests/check.py --v1ref-api-test DIR/build/api_test (the q35 group:
# seqfile/v1_same, seqfile/v2_on_v1_reader). Needs git with the commit; SEQFILE_V1REF_COMMIT: another one
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="${1:?usage: tests/make_seqfile_v1ref.sh DIR}"
REF="${SEQFILE_V1REF_COMMIT:-915894a}"
PY=python3
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
want="$REF $(sed -n '/\/\/ seqfile-out: begin/,/\/\/ seqfile-out: end/p' "$ROOT/tests/api_test.c" | cat - "$0" | sha256sum | cut -c1-16)"
if [[ -x "$D/build/api_test" && -f "$D/stamp" && "$(cat "$D/stamp")" == "$want" ]]; then
    echo "$D/build/api_test ($REF + --seqfile-out / --seqfile-load; up to date)"
    exit 0
fi
rm -rf "$D"
mkdir -p "$D"
git -C "$ROOT" archive "$REF" src tests/api_test.c Makefile | tar -x -C "$D"
"$PY" -I - "$ROOT/tests/api_test.c" "$D/tests/api_test.c" <<'EOF'
import sys
src = open(sys.argv[1]).read()
blk = src[src.index("// seqfile-out: begin"):src.index("// seqfile-out: end")] + "// seqfile-out: end\n\n"
dst = open(sys.argv[2]).read()
anchor = "int main(int argc, char **argv) {\n    int only_digest = 0;\n"
assert dst.count(anchor) == 1 and "seqfile-out" not in dst, "not 915894a's api_test.c"
disp = "".join(f'    if (argc > 1 && !strcmp(argv[1], "--seqfile-{m}")) return seqfile_{m}_main(argc, argv);\n' for m in ("out", "load"))
open(sys.argv[2], "w").write(dst.replace(anchor, blk + anchor + disp))
EOF
make -C "$D" -s -j8 build/api_test
echo "$want" > "$D/stamp"
echo "$D/build/api_test ($REF + --seqfile-out / --seqfile-load)"
