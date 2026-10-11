"""Executable module entry point shared with the installed cv-pipeline command.

Keep the main guard: multiprocessing spawn reimports entry modules in children.
Calling main during that import would recursively launch the CLI and workers.
All orchestration stays in cli.main and PipelineSupervisor, not this wrapper.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
