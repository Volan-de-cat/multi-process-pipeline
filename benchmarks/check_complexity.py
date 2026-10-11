"""Fail CI when a source function/method exceeds the cyclomatic budget of 20.

Classes are containers rather than independent execution paths. Radon still
reports their aggregate metrics in CI artifacts; the gate checks callable bodies.
"""

from pathlib import Path

from radon.complexity import cc_visit


def main() -> int:
    violations = []
    root = Path(__file__).resolve().parents[1]
    paths = [path for folder in ("src", "benchmarks") for path in (root / folder).rglob("*.py")]
    for path in sorted(paths):
        for block in cc_visit(path.read_text()):
            callables = block.methods if hasattr(block, "methods") else [block]
            for item in callables:
                if item.complexity > 20:
                    violations.append(f"{path.name}:{item.lineno} {item.name}: {item.complexity}")
    if violations:
        print("\n".join(violations))
        return 1
    print("All source and benchmark functions/methods have cyclomatic complexity <= 20.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
