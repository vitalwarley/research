import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Generator, Any, List, Union, Callable


Img = np.ndarray  # HxWxC image in numpy (read with cv2.imread)
Tensor = torch.Tensor # HxWxC image in torch 
Embedding = torch.Tensor  # 1x512 image embedding (CNN output given input image)
Batch = torch.Tensor  # NxCxHxW batch of input images
Labels = torch.Tensor  # Nx1 float unscaled kinship relation labels
ImgOrPath = Union[Img, Path]
ImgPairOrPath = Tuple[ImgOrPath, ImgOrPath]
PairPath = Tuple[Path, Path]
