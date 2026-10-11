"""Command-line boundary for validation, overrides and lifecycle delegation.

Parsing and configuration printing do not start processes or load CV libraries.
A finite duration begins after stage readiness. Reporter paths are relative to
the working directory; existing files and symlink components are rejected.
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from .config import config_json, load_config


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv, apply output override, validate configuration and return exit status.

    --print-config exits without starting workers. Duration must be positive and
    finite. Invalid configuration uses argparse status 2; runtime failure is nonzero.
    Import Supervisor lazily so help/config inspection does not require CV startup.
    """
    parser = argparse.ArgumentParser(description="Multi-process camera CV pipeline")
    parser.add_argument("--config", type=Path, help="JSON configuration path")
    parser.add_argument(
        "--duration", type=float, help="Capture duration in seconds after all stages are ready"
    )
    parser.add_argument("--output", help="Write Reporter JSON Lines to this file (default: stdout)")
    parser.add_argument(
        "--print-config", action="store_true", help="Print validated configuration and exit"
    )
    parser.add_argument("--summary", help="Create a fresh final accounting JSON file")
    args = parser.parse_args(argv)
    if args.duration is not None and (
        not math.isfinite(args.duration) or not 0 < args.duration <= 86400
    ):
        parser.error("--duration must be finite and between 0 and 86400 seconds")
    try:
        config = load_config(args.config, output_path=args.output)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(config_json(config))
        return 0
    from .supervisor import PipelineSupervisor

    return PipelineSupervisor(config, args.duration, args.summary).run()
