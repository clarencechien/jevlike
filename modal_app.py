"""Modal entry point: image, volumes, model download, llama-server, benches.

Usage (via scripts/modal.sh so token env vars are mapped):
  scripts/modal.sh run modal_app.py --which download
  scripts/modal.sh run modal_app.py --which probe
  scripts/modal.sh run modal_app.py --which smoke
  scripts/modal.sh run modal_app.py --which latency
  scripts/modal.sh run --detach modal_app.py --which accuracy --args "--resume"
  scripts/modal.sh volume get gb10-decide-results / ./results/modal/
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

MODEL_REPO = "unsloth/gemma-4-26B-A4B-it-GGUF"
MODEL_FILE = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"  # 16.95 GB on HF (checked 2026-09-23)
GPU = os.environ.get("GB10_GPU", "L4")
SERVER_BIN = "/app/llama-server"

app = modal.App("gb10-decide")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=True)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)

image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.11")
    .entrypoint([])  # image's ENTRYPOINT is llama-server; Modal needs a plain shell
    .pip_install("huggingface_hub", "numpy", "requests")
    .add_local_dir("decide", remote_path="/root/decide")
    .add_local_dir("bench", remote_path="/root/bench")
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(image=image, volumes={"/models": models}, timeout=60 * 60)
def download():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir="/models")
    models.commit()
    return {"path": p, "bytes": os.path.getsize(p)}


def _server_version():
    for cand in [SERVER_BIN, "/llama-server", "/usr/local/bin/llama-server"]:
        if os.path.exists(cand):
            out = subprocess.run([cand, "--version"], capture_output=True, text=True)
            return cand, (out.stdout + out.stderr).strip()
    out = subprocess.run(["bash", "-lc", "find / -name 'llama-server' -type f 2>/dev/null | head"], capture_output=True, text=True)
    return None, out.stdout


@app.function(image=image, gpu=GPU, volumes={"/models": models, "/results": results}, timeout=60 * 10)
def probe():
    """M0: image boots on GPU, llama-server binary found, nvidia-smi works. No model needed."""
    bin_path, ver = _server_version()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    have_model = os.path.exists(f"/models/{MODEL_FILE}")
    size = os.path.getsize(f"/models/{MODEL_FILE}") if have_model else 0
    info = {
        "gpu": GPU,
        "nvidia_smi": smi.stdout.strip() or smi.stderr.strip(),
        "server_bin": bin_path,
        "server_version": ver,
        "model_repo": MODEL_REPO,
        "model_file": MODEL_FILE,
        "model_present": have_model,
        "model_bytes": size,
        "modal_version": modal.__version__,
    }
    os.makedirs("/results", exist_ok=True)
    with open("/results/00-env.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    results.commit()
    return info


def start_server(n_parallel=4, ctx=16384, port=8080, extra=()):
    cmd = [
        SERVER_BIN, "-m", f"/models/{MODEL_FILE}",
        "-ngl", "99", "-c", str(ctx), "-np", str(n_parallel),
        "--port", str(port), "--host", "127.0.0.1",
        "--jinja", "--reasoning-budget", "0",
        "--metrics",
        *extra,
    ]
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    t0 = time.time()
    for _ in range(900):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


@app.function(image=image, gpu=GPU, volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_bench(which: str, args: str = "", n_parallel: int = 4, ctx: int = 16384):
    """which in {smoke, latency, accuracy}; writes /results/<which>/..."""
    import sys

    sys.path.insert(0, "/root")
    t_start = time.time()
    proc = start_server(n_parallel=n_parallel, ctx=ctx)
    t_ready = time.time()
    try:
        mod = __import__(f"bench.{which}", fromlist=["main"])
        out_dir = f"/results/{which}"
        os.makedirs(out_dir, exist_ok=True)
        ret = mod.main(base_url="http://127.0.0.1:8080", out_dir=out_dir, args=args, commit=results.commit)
        with open(f"/results/{which}/_run.json", "a") as f:
            f.write(json.dumps({"which": which, "args": args, "gpu": GPU, "n_parallel": n_parallel, "ctx": ctx,
                                "server_start_s": round(t_ready - t_start, 1),
                                "bench_s": round(time.time() - t_ready, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        results.commit()
        return ret
    finally:
        proc.terminate()


@app.local_entrypoint()
def main(which: str = "smoke", args: str = "", n_parallel: int = 4, n_ctx: int = 16384):
    if which == "download":
        print(json.dumps(download.remote(), indent=2))
    elif which == "probe":
        print(json.dumps(probe.remote(), indent=2, ensure_ascii=False))
    else:
        ret = run_bench.remote(which, args, n_parallel, n_ctx)
        print(json.dumps(ret, indent=2, ensure_ascii=False, default=str)[:4000])
