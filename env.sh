# Source me (bash or zsh):   source env.sh
#
# 1. sources the machine toolchain script (conda, NVHPC/CUDA, oneAPI, ...)  <- required before torch/CUDA
# 2. activates the project's conda env (default: myvoice; override with MYVOICE_ENV=...)
# 3. puts the pip-installed NVIDIA libs (cuBLAS, cuDNN 9, cudart) on LD_LIBRARY_PATH so
#    CTranslate2 (faster-whisper) and llama.cpp find them
# 4. forces Hugging Face libraries offline so nothing is downloaded at runtime

if [ -n "${ZSH_VERSION-}" ]; then
  _mv_self="${(%):-%x}"
elif [ -n "${BASH_SOURCE+x}" ]; then
  _mv_self="${BASH_SOURCE[0]}"
else
  echo "env.sh: unsupported shell (need bash or zsh)" >&2
  return 1 2>/dev/null || exit 1
fi
MYVOICE_ROOT="$(cd -- "$(dirname -- "$_mv_self")" && pwd)"
export MYVOICE_ROOT
unset _mv_self

MYVOICE_SETVARS="${MYVOICE_SETVARS:-/home/saleh/00_software/dir_install/setvars.all.sh}"
if [ ! -f "$MYVOICE_SETVARS" ]; then
  echo "env.sh: toolchain script not found: $MYVOICE_SETVARS" >&2
  return 1 2>/dev/null || exit 1
fi
# `source` without arguments hands the *caller's* arguments (e.g. run.sh's "check") to the
# toolchain scripts, which reject unknown arguments. Source them with an empty list instead.
_mv_saved_args=("$@")
set --
if [ -n "${MYVOICE_VERBOSE-}" ]; then
  source "$MYVOICE_SETVARS"
else
  source "$MYVOICE_SETVARS" >/dev/null
fi
set -- "${_mv_saved_args[@]}"
unset _mv_saved_args

MYVOICE_ENV="${MYVOICE_ENV:-myvoice}"
if ! conda activate "$MYVOICE_ENV" 2>/dev/null; then
  echo "env.sh: conda env '$MYVOICE_ENV' not found. Create it with: bash scripts/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi

_mv_nvlibs="$(python - <<'PY' 2>/dev/null
import glob, os, sysconfig
site = sysconfig.get_paths()["purelib"]
dirs = sorted(d for d in glob.glob(os.path.join(site, "nvidia", "*", "lib")) + glob.glob(os.path.join(site, "nvidia", "*", "*", "lib")) if os.path.isdir(d))
print(":".join(dirs))
PY
)"
if [ -n "$_mv_nvlibs" ]; then
  export LD_LIBRARY_PATH="${_mv_nvlibs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _mv_nvlibs

export PYTHONPATH="$MYVOICE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MYVOICE_ENV_SOURCED=1

echo "[+] myvoice: env '$MYVOICE_ENV' active, offline mode on ($MYVOICE_ROOT)"
