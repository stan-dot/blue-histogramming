from argparse import ArgumentParser
from collections.abc import Sequence

from . import __version__
from .main import run_server

__all__ = ["main"]


def main(args: Sequence[str] | None = None) -> None:
    parser = ArgumentParser(
        prog="blue_histogramming",
        description="A package for histogramming data in Python.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server and STOMP listener",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.serve:
        run_server()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
