#!/usr/bin/env python3
"""myvoice: offline Italian <-> English speech translator.

    ./myvoice.py          conversation app, auto mode (translates everything it hears)
    ./myvoice.py -m       conversation app, manual mode (keys start and skip each step)
    ./myvoice.py -h       all commands (file, text, voices, setup, download, check, ...)

Any Python 3 can start it: it loads setvars.all.sh and the `myvoice` conda env by itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from myvoice.launch import main  # noqa: E402

if __name__ == "__main__":
    main()
