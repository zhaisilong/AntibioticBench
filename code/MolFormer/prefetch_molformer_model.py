#!/usr/bin/env python3
"""Download/validate the MoLFormer tokenizer and checkpoint in the configured HF cache."""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="ibm-research/MoLFormer-XL-both-10pct")
    p.add_argument("--model-revision", default="compat-v4")
    p.add_argument("--local-files-only", type=int, choices=[0, 1], default=0)
    args = p.parse_args()

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    common = {
        "trust_remote_code": True,
        "local_files_only": bool(args.local_files_only),
    }
    if args.model_revision:
        common["revision"] = args.model_revision
    print(f"Python={sys.version.split()[0]}")
    print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; CUDA available={torch.cuda.is_available()}")
    print(f"Transformers={transformers.__version__}")
    print(f"Loading tokenizer revision={args.model_revision!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **common)
    kwargs = dict(common)
    kwargs["deterministic_eval"] = True
    kwargs["torch_dtype"] = torch.float32
    try:
        model = AutoModel.from_pretrained(args.model_id, **kwargs)
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
        kwargs.pop("torch_dtype", None)
        kwargs["dtype"] = torch.float32
        model = AutoModel.from_pretrained(args.model_id, **kwargs)
    model.eval()
    sample = ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    batch = tokenizer(sample, padding=True, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**batch)
    print(f"pooler_output shape={tuple(outputs.pooler_output.shape)}")
    print(f"hidden_size={model.config.hidden_size}; layers={model.config.num_hidden_layers}; max_length={model.config.max_position_embeddings}")
    print("MoLFormer cache/environment validation: PASSED")


if __name__ == "__main__":
    main()
