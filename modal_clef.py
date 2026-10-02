"""v10 (docs/handoff-v10-clef.md): Cloudflare Clef / Clef-flash on Modal, in-process transformers (bench/clef_run.py).

Usage (via scripts/modal.sh so token env vars are mapped):
  scripts/modal.sh run modal_clef.py::download --model clef-flash          # CPU container, into the models volume
  scripts/modal.sh run modal_clef.py::download --model clef
  scripts/modal.sh run modal_clef.py::run_l40s --model clef-flash --quant bf16 --suites smoke
  scripts/modal.sh run modal_clef.py::run_h100 --model clef --quant bf16 --suites smoke,latency
  scripts/modal.sh run --detach modal_clef.py::run_l40s --model clef --quant fp8 --suites smoke,d0,...
  scripts/modal.sh volume get gb10-decide-results /clef ./results/modal/clef
Only synthetic data is mounted (data/); weights live in the models volume and are never committed.
"""
import os

import modal

REPOS = {  # pinned HF revisions (2026-10-02)
    "clef-flash": ("Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6"),
    "clef": ("Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
}

app = modal.App("gb10-decide-clef")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=True)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    # torch from the cu128 index so it matches the image's nvcc (PyPI's default 2.11 wheel is cu130)
    .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "packaging", "ninja", "wheel", "setuptools")
    .pip_install("transformers==5.10.2", "accelerate", "safetensors", "huggingface_hub", "pillow", "requests",
                 "flash-linear-attention", "bitsandbytes")
    # causal-conv1d is the other half of the Qwen3.5 fast path; optional (the torch conv1d fallback is fine for prefill)
    .run_commands("pip install --no-build-isolation causal-conv1d || echo 'causal-conv1d not installed'")
    .add_local_dir("decide", remote_path="/root/decide")
    .add_local_dir("bench", remote_path="/root/bench")
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(image=cpu_image, volumes={"/models": models}, timeout=60 * 90, cpu=4)
def download(model: str = "clef-flash"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download

    repo, rev = REPOS[model]
    p = snapshot_download(repo, revision=rev, local_dir=f"/models/clef/{model}")
    models.commit()
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    print({"path": p, "gb": round(size / 1e9, 2)})
    return {"path": p, "gb": round(size / 1e9, 2)}


def _run(model, quant, suites, tag, token_budget):
    import sys

    sys.path.insert(0, "/root")
    from bench import clef_run

    out = f"/results/clef/{tag or model + '-' + quant}"
    s = clef_run.main(f"/models/clef/{model}", quant, out, suites.split(","), token_budget, commit=results.commit)
    results.commit()
    return s


@app.function(image=image, gpu="L40S", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_l40s(model: str = "clef-flash", quant: str = "bf16", suites: str = "smoke", tag: str = "", token_budget: int = 16384):
    return _run(model, quant, suites, tag, token_budget)


@app.function(image=image, gpu="H100", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 2)
def run_h100(model: str = "clef", quant: str = "bf16", suites: str = "smoke", tag: str = "", token_budget: int = 16384):
    return _run(model, quant, suites, tag, token_budget)
