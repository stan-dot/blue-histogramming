"""Interface for ``python -m blue_histogramming``."""

from argparse import ArgumentParser
from collections.abc import Sequence

from . import __version__

__all__ = ["main"]


def main(args: Sequence[str] | None = None) -> None:
    print(f"ARGS: {args}")
    """Argument parser for the CLI."""
    print(
        "blue_histogramming is a package for histogramming data in Python.\n"
        "For more information, please visit"
    )
    parser = ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    parser.parse_args(args)


if __name__ == "__main__":
    main()
