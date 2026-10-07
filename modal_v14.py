"""v14 (docs/handoff-v14-sop-routing.md): 100-SOP routing on Modal.

  scripts/modal.sh run modal_v14.py::llm --name e4b --mode tournament
  scripts/modal.sh run modal_v14.py::llm --name 26b --mode hybrid
  scripts/modal.sh run modal_v14.py::eg_gpu                      # EmbeddingGemma 2 latency on an L4 (ranking must match the CPU run)
  scripts/modal.sh volume get optjev-results v14 results/modal/
LLMs: llama.cpp b11371 binary from the models volume (jevlike v11), same flags as v12. Only synthetic data is mounted.
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

BUILD_DIR = "/models/llama-b11371"
SERVER_BIN = f"{BUILD_DIR}/bin/llama-server"
MODEL_PATH = {"26b": "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", "e4b": "/models/gemma-4-E4B-it-Q8_0.gguf"}
PORT = 8080

app = modal.App("jevlike-v14")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("optjev-results", create_if_missing=True)

llm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([]).apt_install("libgomp1").pip_install("numpy", "requests")
    .add_local_dir("decide", remote_path="/root/decide").add_local_dir("bench", remote_path="/root/bench").add_local_dir("data/v14", remote_path="/root/data/v14")
)
eg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "torchaudio", "sentence-transformers>=6.1.0", "transformers>=5.19.0", "numpy")
    .add_local_dir("bench", remote_path="/root/bench").add_local_dir("data/v14", remote_path="/root/data/v14")
)


def _start(model_path):
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    cmd = [SERVER_BIN, "-m", model_path, "-ngl", "99", "-c", "16384", "-np", "4", "--port", str(PORT), "--host", "127.0.0.1",
           "--jinja", "--reasoning-budget", "0", "--swa-full", "--metrics"]
    proc = subprocess.Popen(cmd); t0 = time.time()
    for _ in range(1200):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); print(f"healthy after {time.time() - t0:.1f}s", flush=True)
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate(); raise RuntimeError("llama-server did not become healthy")


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 2)
def llm(name: str = "26b", mode: str = "tournament", workers: int = 4, limit: int = 0, k: int = 10, subset: str = "", out: str = "v14"):
    """v14: modes tournament / hybrid. v15 (docs/handoff-v15-shortlist-tournament.md): mode shortlist with k = 20/30/50, out = v15."""
    import sys
    sys.path.insert(0, "/root")
    proc = _start(MODEL_PATH[name])
    try:
        from bench import route_bench
        out_dir = f"/results/{out}/{name}"
        args = f"--mode {mode} --workers {workers} --limit {limit} --k {k}" + (f" --subset {subset}" if subset else "")
        ret = route_bench.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=args)
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        tag = mode + (str(k) if mode == "shortlist" else "") + (f"_{subset}" if subset else "")
        json.dump({**ret, "gpu": smi, "model": MODEL_PATH[name], "server": "llama.cpp b11371"}, open(f"{out_dir}/_run_{tag}.json", "w"), indent=1)
        results.commit()
        return ret
    finally:
        proc.terminate()


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 2)
def latency_single(name: str = "26b"):
    """v15: one query at a time (workers 1) on the first 30 test queries, all five configs in one container and one server."""
    import sys
    sys.path.insert(0, "/root")
    proc = _start(MODEL_PATH[name])
    try:
        from bench import route_bench
        out = {}
        for mode, k in (("shortlist", 10), ("shortlist", 20), ("shortlist", 30), ("shortlist", 50), ("tournament", 0)):
            out[f"{mode}{k or ''}"] = route_bench.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=f"/results/v15/{name}",
                                                   args=f"--mode {mode} --k {k or 10} --subset test30 --workers 1")
            results.commit()
        return out
    finally:
        proc.terminate()


@app.function(image=eg_image, gpu="L4", volumes={"/results": results}, timeout=60 * 30)
def eg_gpu():
    import sys
    sys.path.insert(0, "/root")
    from bench import embed_route
    ret = embed_route.main(["--device", "cuda", "--out", "/results/v14/eg_gpu"])
    results.commit()
    return ret
