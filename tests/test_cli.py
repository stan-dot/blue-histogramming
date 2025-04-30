import subprocess
import sys

from blue_histogramming import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "blue_histogramming", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
