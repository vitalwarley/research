import sys
from pathlib import Path

# Add the parent directory to sys.path using pathlib
sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402

from tasks.base import BaseTask


class FaCoRNetTask2(BaseTask):
    operation_name = "facornet:tri_subject_train"


if __name__ == "__main__":
    FaCoRNetTask2.main() 