from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import mxnet as mx
import numpy as np

Img = np.ndarray  # HxWxC image in numpy (read with cv2.imread)
MxImg = (
    mx.nd.NDArray
)  # HxWxC image in mxnet (read with mx.img.imread or converted from Img)
Embedding = mx.nd.NDArray  # 1x512 image embedding (CNN output given input image)
MxImgArray = mx.nd.NDArray  # NxCxHxW batch of input images
Labels = mx.nd.NDArray  # Nx1 float unscaled kinship relation labels
ImgOrPath = Union[Img, Path]
ImgPairOrPath = Tuple[ImgOrPath, ImgOrPath]
PairPath = Tuple[Path, Path]
