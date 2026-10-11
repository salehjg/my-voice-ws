#!/usr/bin/env bash
# Make a CUDA 12.x toolkit usable with glibc >= 2.41 without modifying it.
#
#   bash scripts/patch_cuda_glibc.sh SRC_CUDA_ROOT DST_DIR     # prints DST_DIR when done
#
# glibc 2.41+ declares sinpi/cospi/rsqrt (+ the float versions) as noexcept, while CUDA <= 12.9's
# crt/math_functions.h declares them without it, so every nvcc compile fails with
#   mathcalls.h(85): error: exception specification is incompatible with that of previous function "sinpi"
# (fixed in CUDA 13). DST_DIR becomes a "shadow" toolkit: symlinks to SRC_CUDA_ROOT, except a
# patched copy of that one header (plus real directories along its path, so that the header's
# relative #includes resolve to the patched copy).

set -eo pipefail
SRC="$(cd -- "$1" && pwd)"
DST="$2"
[ -x "$SRC/bin/nvcc" ] || { echo "not a CUDA toolkit: $SRC" >&2; exit 2; }
INC=targets/x86_64-linux/include
HDR="$INC/crt/math_functions.h"

# mirror DIR: make DST/DIR a real directory whose entries are symlinks to SRC/DIR/*
mirror() {
  mkdir -p "$DST/$1"
  local e
  for e in "$SRC/$1"/* "$SRC/$1"/.[!.]*; do
    [ -e "$e" ] || [ -L "$e" ] || continue
    ln -sfn "$e" "$DST/$1/$(basename "$e")"
  done
}

rm -rf "$DST"
mirror .
rm "$DST/targets" && mirror targets
rm "$DST/targets/x86_64-linux" && mirror targets/x86_64-linux
rm "$DST/$INC" && mirror "$INC"
rm "$DST/$INC/crt" && mirror "$INC/crt"
rm "$DST/bin" && mirror bin   # nvcc finds its headers relative to its own (unresolved) path
# include/ and lib64/ are relative symlinks into targets/ in the original; recreate them as such.
ln -sfn "$INC" "$DST/include"
ln -sfn targets/x86_64-linux/lib "$DST/lib64"

rm "$DST/$HDR"
sed -E \
  -e 's/^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ +(double|float) +(sinpi|sinpif|cospi|cospif|rsqrt|rsqrtf)\((double|float) x\));/\1 __MYVOICE_GLIBC_NOEXCEPT;/' \
  "$SRC/$HDR" > "$DST/$HDR"
n=$(grep -c "__MYVOICE_GLIBC_NOEXCEPT;" "$DST/$HDR" || true)
[ "$n" = 6 ] || { echo "expected 6 declarations to patch in $HDR, found $n" >&2; exit 1; }
# Define the macro at the top: noexcept in C++ (matches glibc's __THROW), nothing in C.
sed -i '0,/^#if !defined(__MATH_FUNCTIONS_H__)/s//#ifndef __MYVOICE_GLIBC_NOEXCEPT\n#ifdef __cplusplus\n#define __MYVOICE_GLIBC_NOEXCEPT noexcept (true)\n#else\n#define __MYVOICE_GLIBC_NOEXCEPT\n#endif\n#endif\n&/' "$DST/$HDR"
grep -q "define __MYVOICE_GLIBC_NOEXCEPT noexcept" "$DST/$HDR" || { echo "could not insert the macro into $HDR" >&2; exit 1; }
echo "$DST"
