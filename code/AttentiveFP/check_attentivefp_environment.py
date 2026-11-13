#!/usr/bin/env python3
from __future__ import annotations
import json, platform, sys
EXPECTED = 897_201
try:
    import numpy as np
    import pandas as pd
    import scipy, sklearn, torch, torch_geometric
    from rdkit import rdBase
    from torch_geometric.nn.models import AttentiveFP
except Exception as exc:
    raise SystemExit(f"Environment check failed: {type(exc).__name__}: {exc}") from exc
m = AttentiveFP(43, 200, 1, 14, 2, 2, 0.20)
params = sum(p.numel() for p in m.parameters() if p.requires_grad)
x = torch.zeros((3, 43)); edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
edge_attr = torch.zeros((2, 14)); batch = torch.tensor([0, 0, 1], dtype=torch.long)
with torch.no_grad(): y = m(x, edge_index, edge_attr, batch)
payload = {
 "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
 "torch_geometric": torch_geometric.__version__, "rdkit": rdBase.rdkitVersion,
 "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
 "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
 "attentivefp_trainable_params": params, "expected_trainable_params": EXPECTED,
 "parameter_count_matches": params == EXPECTED, "smoke_test_output_shape": list(y.shape),
 "smoke_test_finite": bool(torch.isfinite(y).all().item())}
print(json.dumps(payload, ensure_ascii=False, indent=2))
if params != EXPECTED: raise SystemExit(f"Parameter-count mismatch: expected {EXPECTED:,}, got {params:,}")
if list(y.shape) != [2, 1] or not torch.isfinite(y).all(): raise SystemExit("AttentiveFP smoke test failed")
