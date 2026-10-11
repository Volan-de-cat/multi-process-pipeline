"""Multi-process camera pipeline package with no startup side effects.

Importing the package creates no camera, process, thread or shared-memory mapping.
The CLI is the composition entry point; individual adapters and policy helpers
remain importable for tests without accidentally starting the application.
"""
