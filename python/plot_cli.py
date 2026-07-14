from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


def add_display_options(
    parser: argparse.ArgumentParser,
    *,
    save_png: bool = True,
) -> None:
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Construct figures and exit without opening a blocking GUI window.",
    )
    if save_png:
        parser.add_argument(
            "--save-png",
            type=Path,
            default=None,
            help=(
                "Optional PNG output path. If multiple figures are open, PATH "
                "is treated as a filename prefix or an output directory."
            ),
        )


def install_noop_show_when(condition: bool) -> Callable[[], None]:
    if not condition:
        return lambda: None

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None

    def restore() -> None:
        plt.show = original_show

    return restore


def _figure_output_path(base_path: Path, index: int, count: int) -> Path:
    if count == 1 and base_path.suffix:
        path = base_path
    elif base_path.suffix:
        path = base_path.with_name(f"{base_path.stem}_{index:02d}.png")
    else:
        path = base_path / f"figure_{index:02d}.png"

    if path.suffix.lower() != ".png":
        raise ValueError(f"--save-png expects a .png path, got '{path}'")
    return path


def save_open_figures(save_png: Path | str | None) -> list[Path]:
    if save_png is None:
        return []

    base_path = Path(save_png)
    if base_path.suffix and base_path.suffix.lower() != ".png":
        raise ValueError(f"--save-png expects a .png path, got '{base_path}'")

    figures = [plt.figure(num) for num in plt.get_fignums()]
    saved_paths: list[Path] = []
    for index, fig in enumerate(figures):
        path = _figure_output_path(base_path, index, len(figures))
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"dpi": 160, "bbox_inches": "tight"}
        pad_inches = getattr(fig, "_adappfem_savefig_pad_inches", None)
        if pad_inches is not None:
            save_kwargs["pad_inches"] = float(pad_inches)
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
    return saved_paths


def finish_figures(
    *,
    show: bool,
    save_png: Path | str | None = None,
) -> list[Path]:
    saved_paths = save_open_figures(save_png)
    if show:
        plt.show()
    else:
        plt.close("all")
    return saved_paths
