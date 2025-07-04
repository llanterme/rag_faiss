"""Launcher script for Streamlit UI."""

import os
import sys
from pathlib import Path


def main():
    """Launch the Streamlit application.

    This function serves as an entry point for the Poetry script.
    It executes the 'streamlit run' command on the UI module.
    """
    # Get the path to the ui.py file
    script_path = Path(__file__).parent / "ui.py"

    # Prepare the command
    cmd = f"streamlit run {script_path.absolute()}"

    # Execute the command
    os.system(cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
