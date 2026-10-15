#!/usr/bin/env bash
# Repeat a pytest invocation in fresh processes to sample BLAS rounding luck.
# Usage: stress_loop.sh <repeats> <jitter 0|1> <pytest paths...>
set -u

repeats=$1
jitter=$2
shift 2

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
ulimit -s 20000

mkdir -p pyscftmpdir
cat > .pyscf_conf.py <<'EOF'
pbc_tools_pbc_fft_engine = "NUMPY+BLAS"
scf_hf_SCF_mute_chkfile = True
TMPDIR = "./pyscftmpdir"
EOF

fails=0
summary=()
for i in $(seq 1 "$repeats"); do
    if [ "$jitter" = "1" ]; then
        n=$(( (RANDOM % 4) + 1 ))
        export OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n
    else
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    fi
    log=$(mktemp)
    if pytest -q -c pytest.ini "$@" >"$log" 2>&1; then
        status=ok
    else
        status=FAIL
        fails=$((fails + 1))
        cat "$log"
    fi
    rm -f "$log"
    line="iter $i/$repeats [threads=$OMP_NUM_THREADS]: $status"
    echo "$line"
    summary+=("$line")
done
rm -rf pyscftmpdir

{
    echo "## CABS stress summary"
    echo ""
    for line in "${summary[@]}"; do
        echo "- $line"
    done
    echo ""
    echo "Failed iterations: $fails / $repeats"
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}"

echo "stress: $fails/$repeats failed"
[ "$fails" -eq 0 ]
