"""Import each package module named on the command line; exit 1 if any fails."""

import importlib
import sys
import traceback
from pathlib import Path


def main(paths: list[str]) -> int:
    failed = False
    for path in paths:
        module = ".".join(Path(path).with_suffix("").parts)
        try:
            importlib.import_module(module.removesuffix(".__init__"))
        except Exception:
            failed = True
            sys.stderr.write(f"{path}\n{traceback.format_exc()}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
