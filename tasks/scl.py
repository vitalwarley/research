import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tasks.base import BaseTask # noqa: E402


class SCL(BaseTask):
    operation_name = "scl:train"


if __name__ == "__main__":
    SCL.main()
