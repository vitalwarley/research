import sys
from pathlib import Path

# Add the parent directory to sys.path using pathlib
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Why isn't sufficient to do it in base?

from tasks.base import base_main  # noqa


def main(args=None):
    base_main(args=args, operation_name="facornet")


if __name__ == "__main__":
    main()
