from __future__ import annotations

from pathlib import Path


def repository_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def generator_hint(generator_test: str) -> str:
    return (
        "Generate or copy the matching artifact before plotting. "
        f"Missing generator/artifact name: {generator_test}."
    )


def require_generator_file(path: Path, generator_test: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required artifact '{path}'. {generator_hint(generator_test)}"
        )

    return path


def require_generator_directory(path: Path, generator_test: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing required artifact directory '{path}'. {generator_hint(generator_test)}"
        )

    return path
