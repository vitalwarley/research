import sys
from pathlib import Path

# Add the parent directory to sys.path using pathlib
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Why isn't sufficient to do it in base?

from tasks.base import BaseTask  # noqa


class FacorNet(BaseTask):
    operation_name = "facornet:train"


if __name__ == "__main__":
    FacorNet.main()
