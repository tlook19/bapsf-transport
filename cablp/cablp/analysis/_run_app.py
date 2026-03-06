"""
Entry point for the LAPDSim Streamlit GUI.

Run with:
    poetry run lapd-app
or directly:
    streamlit run /path/to/app.py
"""
import subprocess
import sys
import pathlib

try:
    import setproctitle
    setproctitle.setproctitle("lapd-app")
except ImportError:
    pass


def main():
    app = pathlib.Path(__file__).parent / "app.py"
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app)] + sys.argv[1:]
        )
    )
