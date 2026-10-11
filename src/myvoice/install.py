"""``./myvoice.py setup``: create the ``myvoice`` conda env and install every package (~5 GB).

Standard library only, because it runs before the env exists. Models are a separate step:
``./myvoice.py download``. Running it again reuses the env, skips packages that are already
installed and rebuilds llama-cpp-python.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .launch import ENV_NAME, ROOT, LaunchError, toolchain_env

TORCH_VERSION = "2.9.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
LLAMA_CPP_VERSION = "0.3.36"
LLAMA_CPP_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu125"
NVHPC = Path(os.environ.get("MYVOICE_NVHPC", "/home/saleh/00_software/dir_install/nvhpc25.7/Linux_x86_64/25.7"))


def step(msg: str) -> None:
    print(f"\n\033[1m==> {msg}\033[0m", flush=True)


def run(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run([str(c) for c in cmd], env=env, check=True)


# --------------------------------------------------------------------------- CUDA vs glibc


def patch_cuda_glibc(src: Path, dst: Path) -> Path:
    """Make a CUDA 12.x toolkit usable with glibc >= 2.41 without modifying it; returns ``dst``.

    glibc 2.41+ declares sinpi/cospi/rsqrt (+ the float versions) as noexcept, while CUDA <= 12.9's
    crt/math_functions.h declares them without it, so every nvcc compile fails with
      mathcalls.h(85): error: exception specification is incompatible with that of previous function "sinpi"
    (fixed in CUDA 13). ``dst`` becomes a "shadow" toolkit: symlinks to ``src``, except a patched
    copy of that one header (plus real directories along its path, so that the header's relative
    #includes resolve to the patched copy).
    """
    src = Path(os.path.abspath(src))
    if not (src / "bin" / "nvcc").is_file():
        raise LaunchError(f"not a CUDA toolkit: {src}")
    inc = "targets/x86_64-linux/include"
    header = f"{inc}/crt/math_functions.h"

    def mirror(rel: str) -> None:
        """Make dst/rel a real directory whose entries are symlinks to src/rel/*."""
        d = dst / rel
        if d.is_symlink():
            d.unlink()
        d.mkdir(parents=True, exist_ok=True)
        for entry in (src / rel).iterdir():
            link = d / entry.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(entry)

    shutil.rmtree(dst, ignore_errors=True)
    # nvcc finds its headers relative to its own (unresolved) path, hence bin/ too
    for rel in (".", "targets", "targets/x86_64-linux", inc, f"{inc}/crt", "bin"):
        mirror(rel)
    # include/ and lib64/ are relative symlinks into targets/ in the original; recreate them as such
    for name, target in (("include", inc), ("lib64", "targets/x86_64-linux/lib")):
        (dst / name).unlink()
        (dst / name).symlink_to(target)

    decl = re.compile(r"^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ +(double|float) +"
                      r"(sinpi|sinpif|cospi|cospif|rsqrt|rsqrtf)\((double|float) x\));", re.M)
    text, n = decl.subn(r"\1 __MYVOICE_GLIBC_NOEXCEPT;", (src / header).read_text(encoding="latin-1"))
    if n != 6:
        raise LaunchError(f"expected 6 declarations to patch in {header}, found {n}")
    # Define the macro at the top: noexcept in C++ (matches glibc's __THROW), nothing in C.
    guard = "#if !defined(__MATH_FUNCTIONS_H__)"
    macro = ("#ifndef __MYVOICE_GLIBC_NOEXCEPT\n#ifdef __cplusplus\n#define __MYVOICE_GLIBC_NOEXCEPT noexcept (true)\n"
             "#else\n#define __MYVOICE_GLIBC_NOEXCEPT\n#endif\n#endif\n")
    at = text.find("\n" + guard)
    if at < 0:
        raise LaunchError(f"could not insert the macro into {header}")
    text = text[: at + 1] + macro + text[at + 1 :]
    (dst / header).unlink()
    (dst / header).write_text(text, encoding="latin-1")
    return dst


# --------------------------------------------------------------------------- steps


def ensure_conda_env(base: dict[str, str]) -> None:
    conda = base.get("CONDA_EXE") or shutil.which("conda", path=base.get("PATH"))
    if not conda:
        raise LaunchError("conda not found after sourcing the toolchain script")
    out = subprocess.run([conda, "env", "list", "--json"], env=base, capture_output=True, text=True, check=True).stdout
    if any(Path(p).name == ENV_NAME for p in json.loads(out)["envs"]):
        step(f"conda env '{ENV_NAME}' already exists, reusing it")
        return
    step(f"creating conda env '{ENV_NAME}' (python 3.12)")
    run([conda, "create", "-y", "-n", ENV_NAME, "-c", "conda-forge", "python=3.12", "pip"], base)


def build_llama(pip: list[str], env: dict[str, str]) -> None:
    """llama-cpp-python compiled for this CPU + GPU. The prebuilt CUDA wheel's CPU code uses
    AVX-512, which this laptop's i9-13950HX lacks (crashes with "illegal hardware instruction")."""
    step(f"building llama-cpp-python {LLAMA_CPP_VERSION} from source (CUDA 12.9 from NVHPC, sm_89, this CPU; ~5-15 min)")
    # CUDA 12.9's math headers clash with glibc >= 2.41 (Arch has 2.44): build against a shadow copy
    # of the toolkit in build/ with one patched header. The NVHPC install itself is not touched.
    cuda = patch_cuda_glibc(NVHPC / "cuda" / "12.9", ROOT / "build" / "cuda-12.9-glibc")
    math = NVHPC / "math_libs" / "12.9"
    env = dict(env)
    env.update(
        # nvcc 12.9 rejects GCC > 14, so compile with the system gcc-13
        CC="/usr/bin/gcc-13", CXX="/usr/bin/g++-13", CUDACXX=str(cuda / "bin" / "nvcc"),
        CPATH=os.pathsep.join([str(math / "include")] + ([env["CPATH"]] if env.get("CPATH") else [])),
        CMAKE_ARGS=" ".join([
            "-DGGML_CUDA=on", "-DCMAKE_CUDA_ARCHITECTURES=89", "-DGGML_NATIVE=on",
            "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13",
            f"-DCUDAToolkit_ROOT={cuda}",
            f"-DCMAKE_LIBRARY_PATH={math / 'lib64'}",
            f"-DCUDAToolkit_CUBLAS_INCLUDE_DIR={math / 'include'}",
            f"-DCMAKE_CUDA_FLAGS=-I{math / 'include'}",
        ]),
        FORCE_CMAKE="1",
        CMAKE_BUILD_PARALLEL_LEVEL=str(len(os.sched_getaffinity(0))),
    )
    # --no-deps: rebuild only llama-cpp-python, leave numpy etc. as they are
    run(pip + ["install", "--force-reinstall", "--no-deps", "--no-binary", "llama-cpp-python",
               f"llama-cpp-python=={LLAMA_CPP_VERSION}"], env)
    run(pip + ["install", f"llama-cpp-python=={LLAMA_CPP_VERSION}"], env)  # its dependencies, in a fresh env


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="./myvoice.py setup",
        description="Create the `myvoice` conda env and install every package (downloads ~5 GB). "
        "Models are separate: ./myvoice.py download.",
    )
    parser.add_argument("--with-transformers", action="store_true", help="also install the transformers/bitsandbytes backend")
    parser.add_argument("--llama-prebuilt", action="store_true",
                        help="prebuilt llama-cpp-python CUDA wheel instead of compiling it (only for AVX-512 CPUs)")
    args = parser.parse_args(argv)
    try:
        # 1. toolchain environment first (conda, CUDA, ...): required before anything else
        ensure_conda_env(toolchain_env())
        env = toolchain_env(ENV_NAME)
        pip = [str(Path(env["CONDA_PREFIX"], "bin", "python")), "-m", "pip", "--no-cache-dir"]  # the disk is nearly full

        # 2. PyTorch (CUDA 12.9 build; pulls nvidia-cublas-cu12 / nvidia-cudnn-cu12 9.x used by CTranslate2)
        step(f"installing torch {TORCH_VERSION} (cu129)")
        run(pip + ["install", "--upgrade", "pip"], env)
        run(pip + ["install", f"torch=={TORCH_VERSION}", "--index-url", TORCH_INDEX], env)

        step("installing core requirements")
        run(pip + ["install", "-r", ROOT / "requirements.txt"], env)

        if args.llama_prebuilt:
            step(f"installing prebuilt llama-cpp-python {LLAMA_CPP_VERSION} (CUDA 12.5 wheel; needs an AVX-512 CPU)")
            run(pip + ["install", f"llama-cpp-python=={LLAMA_CPP_VERSION}", "--only-binary=llama-cpp-python",
                       "--extra-index-url", LLAMA_CPP_INDEX], env)
        else:
            build_llama(pip, env)

        if args.with_transformers:
            step("installing transformers backend")
            run(pip + ["install", "-r", ROOT / "requirements-transformers.txt"], env)
    except (LaunchError, subprocess.CalledProcessError, KeyError) as e:
        print(f"\nsetup failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    step("checking (model files show as missing until you download them)")
    subprocess.run([sys.executable, str(ROOT / "myvoice.py"), "check"])
    step("done. Next: ./myvoice.py download")
    return 0
