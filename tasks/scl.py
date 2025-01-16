import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tasks.base import base_main  # noqa


def main(args=None):
    base_main(args=args, operation_name="scl")


if __name__ == "__main__":
    main()
