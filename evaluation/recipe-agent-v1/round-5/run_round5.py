from __future__ import annotations

import importlib.util
import sys

from pathlib import Path


def _load_round4_runner_main():
    round4_script = Path(__file__).resolve().parents[1] / "round-4" / "run_round4.py"
    spec = importlib.util.spec_from_file_location("round4_runner", round4_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load round-4 runner module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def _has_arg(argv: list[str], arg_name: str) -> bool:
    return any(part == arg_name or part.startswith(f"{arg_name}=") for part in argv)


def main() -> int:
    argv = list(sys.argv[1:])
    script_root = Path(__file__).resolve().parent
    if not _has_arg(argv, "--output-dir"):
        argv.extend(["--output-dir", str(script_root)])
    if not _has_arg(argv, "--round-name"):
        argv.extend(["--round-name", script_root.name])
    if not _has_arg(argv, "--scratch-root"):
        argv.extend(["--scratch-root", r"C:\fmo-r5"])

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        round4_main = _load_round4_runner_main()
        return int(round4_main())
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
