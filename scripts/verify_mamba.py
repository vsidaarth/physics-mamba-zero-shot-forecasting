"""Verify that Mamba executes forward and backward passes on the GPU."""

from __future__ import annotations

import json

import torch
from mamba_ssm import Mamba2


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"

    device = torch.device("cuda")
    model = Mamba2(
        d_model=64,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=32,
    ).to(device)
    inputs = torch.randn(8, 256, 64, device=device, requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = outputs.square().mean()
    loss.backward()
    torch.cuda.synchronize()

    result = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "mamba_block": "Mamba2",
        "input_shape": list(inputs.shape),
        "output_shape": list(outputs.shape),
        "loss": float(loss.detach()),
        "gradient_finite": bool(torch.isfinite(inputs.grad).all()),
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
