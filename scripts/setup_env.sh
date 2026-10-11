#!/usr/bin/env bash
# Create the `myvoice` conda env and install every Python package (downloads ~5 GB).
# Models are NOT downloaded here; see to-download.md / scripts/download_models.sh.
#
#   bash scripts/setup_env.sh                       # default: llama-cpp-python compiled for this CPU + GPU
#   bash scripts/setup_env.sh --with-transformers   # + transformers/bitsandbytes backend
#   bash scripts/setup_env.sh --llama-prebuilt      # prebuilt CUDA wheel instead (only for AVX-512 CPUs)
#
# Env vars: MYVOICE_ENV (env name, default myvoice), MYVOICE_SETVARS (toolchain script).

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${MYVOICE_ENV:-myvoice}"
SETVARS="${MYVOICE_SETVARS:-/home/saleh/00_software/dir_install/setvars.all.sh}"

TORCH_VERSION="2.9.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
LLAMA_CPP_VERSION="0.3.36"
LLAMA_CPP_INDEX="https://abetlen.github.io/llama-cpp-python/whl/cu125"

WITH_TRANSFORMERS=0
LLAMA_FROM_SOURCE=1
for arg in "$@"; do
  case "$arg" in
    --with-transformers) WITH_TRANSFORMERS=1 ;;
    --llama-from-source) LLAMA_FROM_SOURCE=1 ;;
    --llama-prebuilt) LLAMA_FROM_SOURCE=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# 1. Toolchain environment first (conda, CUDA, ...). Must be sourced before anything else.
#    `set --` first: the toolchain scripts would otherwise see (and reject) our own options.
set --
# shellcheck disable=SC1090
source "$SETVARS" || { echo "failed to source $SETVARS" >&2; exit 1; }
set -eo pipefail

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# 2. Conda env
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  step "conda env '$ENV_NAME' already exists, reusing it"
else
  step "creating conda env '$ENV_NAME' (python 3.12)"
  conda create -y -n "$ENV_NAME" -c conda-forge python=3.12 pip
fi
conda activate "$ENV_NAME"
PIP="python -m pip --no-cache-dir"   # no cache: the disk is nearly full

# 3. PyTorch (CUDA 12.9 build; pulls nvidia-cublas-cu12 / nvidia-cudnn-cu12 9.x used by CTranslate2)
step "installing torch $TORCH_VERSION (cu129)"
$PIP install --upgrade pip
$PIP install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"

# 4. Core packages
step "installing core requirements"
$PIP install -r "$ROOT/requirements.txt"

# 5. llama-cpp-python with CUDA. Built locally by default: the prebuilt CUDA wheel's CPU code uses
#    AVX-512, which this laptop's i9-13950HX lacks (crashes with "illegal hardware instruction").
if [ "$LLAMA_FROM_SOURCE" = 1 ]; then
  step "building llama-cpp-python $LLAMA_CPP_VERSION from source (CUDA 12.9 from NVHPC, sm_89, this CPU; ~5-15 min)"
  NV=/home/saleh/00_software/dir_install/nvhpc25.7/Linux_x86_64/25.7
  # CUDA 12.9's math headers clash with glibc >= 2.41 (Arch has 2.44): build against a shadow copy
  # of the toolkit in build/ with one patched header. The NVHPC install itself is not touched.
  CUDA_ROOT="$(bash "$ROOT/scripts/patch_cuda_glibc.sh" "$NV/cuda/12.9" "$ROOT/build/cuda-12.9-glibc")"
  # nvcc 12.9 rejects GCC > 14, so compile with the system gcc-13.
  export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
  export CUDACXX="$CUDA_ROOT/bin/nvcc"
  export CPATH="$NV/math_libs/12.9/include${CPATH:+:$CPATH}"
  export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=on \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DCUDAToolkit_ROOT=$CUDA_ROOT \
    -DCMAKE_LIBRARY_PATH=$NV/math_libs/12.9/lib64 \
    -DCUDAToolkit_CUBLAS_INCLUDE_DIR=$NV/math_libs/12.9/include \
    -DCMAKE_CUDA_FLAGS=-I$NV/math_libs/12.9/include"
  export FORCE_CMAKE=1 CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  # --no-deps: rebuild only llama-cpp-python, leave numpy etc. as they are.
  $PIP install --force-reinstall --no-deps --no-binary llama-cpp-python "llama-cpp-python==$LLAMA_CPP_VERSION"
  $PIP install "llama-cpp-python==$LLAMA_CPP_VERSION"  # its dependencies, if this is a fresh env
else
  step "installing prebuilt llama-cpp-python $LLAMA_CPP_VERSION (CUDA 12.5 wheel; needs an AVX-512 CPU)"
  $PIP install "llama-cpp-python==$LLAMA_CPP_VERSION" \
    --only-binary=llama-cpp-python --extra-index-url "$LLAMA_CPP_INDEX"
fi

# 6. Optional transformers backend
if [ "$WITH_TRANSFORMERS" = 1 ]; then
  step "installing transformers backend"
  $PIP install -r "$ROOT/requirements-transformers.txt"
fi

# 7. Verify (no model loading, no downloads)
step "running ./run.sh check (model files will show as missing until you download them)"
"$ROOT/run.sh" check || true

step "done. Next: download the models (to-download.md, step 2)."
