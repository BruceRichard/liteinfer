from __future__ import annotations

from liteinfer.kernel import test_tensor as _run_test_tensor
from liteinfer.utils import call_if_main
import torch

@call_if_main(__name__)
def test_tensor_kernel():
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")

    result = _run_test_tensor(x, y)
    assert result is None
