#!/usr/bin/env bash
# Run the translator with the right environment:   ./run.sh <command> [options]
#   ./run.sh check | devices | live | file IN.wav -o OUT.wav | text "Ciao!" | selftest
# (env.sh sources setvars.all.sh, activates the conda env and enables offline mode)
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh" >/dev/null || exit 1
exec python -m myvoice "$@"
