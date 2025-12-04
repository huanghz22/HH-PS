"""
Utilities for attorch kernels and layers.
"""


from typing import List, Optional

import torch
import triton

def allow_tf32() -> bool:
    """
    Returns whether the current GPU architecture supports TF32.
    """
    return torch.cuda.get_device_capability()[0] >= 8


def get_n_stages(n_stages: int = 2) -> int:
    """
    Receives number of stages for software pipelining and returns it as-is
    if the GPU architecture is Ampere or newer and 2 otherwise.
    """
    return 2 if torch.cuda.get_device_capability()[0] < 8 else n_stages

