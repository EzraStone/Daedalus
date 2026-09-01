#!/usr/bin/env bash
# The smallest end-to-end fidelity run: a handful of cases against a live server.
#
# The POSIX counterpart of smoke.ps1. It fails when nothing was checked and
# when anything was unreachable, because a report of "0 checked, 0 disagreed"
# is 100% agreement by arithmetic and evidence of nothing at all.

set -euo pipefail
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

cases=10
seed=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --cases) cases="${2:?--cases needs a number}"; shift 2 ;;
        --cases=*) cases="${1#*=}"; shift ;;
        --seed) seed="${2:?--seed needs a number}"; shift 2 ;;
        --seed=*) seed="${1#*=}"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

repository="$(repository_dir)"
python="${PYTHON:-}"
if [ -z "$python" ]; then
    for candidate in "${repository}/.venv/bin/python" python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
            python="$candidate"
            break
        fi
    done
fi
if [ -z "$python" ]; then
    echo "no python found; set \$PYTHON or create .venv" >&2
    exit 1
fi

report="$(harness_dir)/runtime/smoke-report.json"
mkdir -p "$(dirname "$report")"
"$python" "${repository}/harness/compare.py" --cases "$cases" --seed "$seed" --out "$report"

"$python" - "$report" <<'CHECK'
import json
import sys

report = json.loads(open(sys.argv[1], encoding="utf-8").read())
checked, unreachable = report["checked"], report["unreachable"]
if checked <= 0:
    raise SystemExit("fidelity smoke test did not produce any runnable circuits")
if unreachable:
    raise SystemExit(f"fidelity smoke test could not reach {unreachable} cases")
print(f"Smoke agreement: {report['agreed']}/{checked} ({report['agreement']})")
CHECK
