#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi

# AMD's wheels expose HIP through torch.cuda, so the application continues to
# use the standard PyTorch `cuda` device name while executing on ROCm.
uv pip install --python .venv/bin/python \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  'torch==2.11.0+rocm7.13.0' \
  'torchvision==0.26.0+rocm7.13.0'

uv pip install --python .venv/bin/python -r requirements-rocm.txt

.venv/bin/python - <<'PY'
import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn((256, 256), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print("ROCm matrix smoke test: OK")
PY