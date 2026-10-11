"""Start-up, standard library only: ``./myvoice.py`` lands here.

The app needs the machine's toolchain environment (``setvars.all.sh``: conda, CUDA, ...) and the
project's conda env. This module loads both and re-runs the command inside that env, with the
pip-installed NVIDIA libraries (cuBLAS, cuDNN 9) on ``LD_LIBRARY_PATH`` for CTranslate2 and
llama.cpp, and Hugging Face libraries forced offline. ``setup`` runs right here instead, because
it is what creates the env.

Env vars: ``MYVOICE_SETVARS`` (toolchain script), ``MYVOICE_ENV`` (conda env name, default myvoice).
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETVARS = os.environ.get("MYVOICE_SETVARS", "/home/saleh/00_software/dir_install/setvars.all.sh")
ENV_NAME = os.environ.get("MYVOICE_ENV", "myvoice")
READY = "MYVOICE_ENV_READY"  # set in the environment the app runs in


class LaunchError(RuntimeError):
    pass


def toolchain_env(conda_env: str | None = None) -> dict[str, str]:
    """The environment after ``source setvars.all.sh`` (and ``conda activate conda_env``).

    The toolchain script is a shell script, so a bash process sources it and prints the result.
    """
    if not Path(SETVARS).is_file():
        raise LaunchError(f"toolchain script not found: {SETVARS} (set MYVOICE_SETVARS)")
    # $0 must not be the script's own path (it refuses to run unless sourced), and `set --` keeps
    # our arguments away from it (the toolchain scripts reject arguments they don't know).
    script = 'f="$1"; e="$2"; set --; source "$f" >/dev/null || exit 97; '
    if conda_env:
        script += 'conda activate "$e" >/dev/null || exit 98; '
    script += "exec env -0"
    p = subprocess.run(["bash", "-c", script, "myvoice", SETVARS, conda_env or ""], capture_output=True)
    if p.returncode == 98:
        raise LaunchError(f"conda env '{conda_env}' not found; create it with: ./myvoice.py setup")
    if p.returncode != 0:
        tail = p.stderr.decode(errors="replace")[-2000:]
        raise LaunchError(f"sourcing {SETVARS} failed (exit {p.returncode})\n{tail}")
    return dict(item.split("=", 1) for item in os.fsdecode(p.stdout).split("\0") if "=" in item)


def app_env() -> tuple[str, dict[str, str]]:
    """(python executable, environment) to run the app in the conda env."""
    env = toolchain_env(ENV_NAME)
    prefix = env.get("CONDA_PREFIX", "")
    python = Path(prefix, "bin", "python")
    if not python.is_file():
        raise LaunchError(f"no python in conda env '{ENV_NAME}' ({prefix}); repair it with: ./myvoice.py setup")
    libs: list[str] = []
    for site in glob.glob(f"{prefix}/lib/python3*/site-packages"):
        libs += glob.glob(f"{site}/nvidia/*/lib") + glob.glob(f"{site}/nvidia/*/*/lib")
    paths = sorted(d for d in libs if os.path.isdir(d))
    env["LD_LIBRARY_PATH"] = ":".join(paths + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", MYVOICE_ROOT=str(ROOT))
    env[READY] = "1"
    return str(python), env


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if os.environ.get(READY) == "1":  # already inside the env
        from .cli import main as cli_main

        cli_main(argv)
        return
    try:
        if argv[:1] == ["setup"]:
            from .install import main as setup_main

            sys.exit(setup_main(argv[1:]))
        python, env = app_env()
    except LaunchError as e:
        print(f"myvoice: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    os.execve(python, [python, "-m", "myvoice", *argv], env)
