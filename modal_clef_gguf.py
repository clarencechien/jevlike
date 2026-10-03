"""v11 (docs/handoff-v11-clef-llamacpp.md): llama.cpp built from tag b11371 (= 99b9548, Clef support) on Modal.

Usage (via scripts/modal.sh so token env vars are mapped):
  scripts/modal.sh run modal_clef_gguf.py::download --name flash-bf16      # CPU container, into the models volume
  scripts/modal.sh run modal_clef_gguf.py::run_l40s --name flash-bf16 --args "--backend systemone --suites smoke"
  scripts/modal.sh run modal_clef_gguf.py::run_h100 --name clef-bf16 --args "--backend systemone --suites smoke"
  scripts/modal.sh run modal_clef_gguf.py::run_l4 --name 26b --which accuracy --args "--out-sub b11371/D0"   # 26B on the new build
  scripts/modal.sh volume get gb10-decide-results /v11 ./results/modal/v11
Only synthetic data is mounted (data/). There are no Linux CUDA release binaries of llama.cpp, so it is compiled here
(CPU image build); CUDA archs 89 (L4, L40S) and 90 (H100).
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

LLAMA_TAG = "b11371"
LLAMA_COMMIT = "99b95488cac0f00ce3f05af113a8c1e287753f87"
GGUF = {  # name -> (repo, revision, file); pinned 2026-10-03
    "flash-bf16": ("ggml-org/Clef-Flash-GGUF", "4a7a08c09bc63baf043b62b5ba89dd67a0357d95", "Clef-Flash-BF16.gguf"),
    "flash-q8": ("ggml-org/Clef-Flash-GGUF", "4a7a08c09bc63baf043b62b5ba89dd67a0357d95", "Clef-Flash-Q8_0.gguf"),
    "flash-q4": ("ggml-org/Clef-Flash-GGUF", "4a7a08c09bc63baf043b62b5ba89dd67a0357d95", "Clef-Flash-Q4_K_M.gguf"),
    "clef-bf16": ("ggml-org/Clef-GGUF", "5f70656b6670c65eb85ad07a11efe211b5f211bd", "Clef-BF16.gguf"),
    "clef-q8": ("ggml-org/Clef-GGUF", "5f70656b6670c65eb85ad07a11efe211b5f211bd", "Clef-Q8_0.gguf"),
    "clef-q4": ("ggml-org/Clef-GGUF", "5f70656b6670c65eb85ad07a11efe211b5f211bd", "Clef-Q4_K_M.gguf"),
}
OURS_26B = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"  # already in the models volume root (modal_app.py)

app = modal.App("gb10-decide-v11")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=True)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "cmake", "build-essential", "libgomp1")
    .run_commands(
        f"git clone --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp /src/llama.cpp",
        f"cd /src/llama.cpp && test \"$(git rev-parse HEAD)\" = {LLAMA_COMMIT}",
        "cd /src/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='89;90' -DLLAMA_CURL=OFF "
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release",
        "cd /src/llama.cpp && cmake --build build --target llama-server -j $(nproc)",
    )
    .pip_install("numpy", "requests", "huggingface_hub")
    .add_local_dir("decide", remote_path="/root/decide")
    .add_local_dir("bench", remote_path="/root/bench")
    .add_local_dir("data", remote_path="/root/data")
)
SERVER_BIN = "/src/llama.cpp/build/bin/llama-server"


@app.function(image=cpu_image, volumes={"/models": models}, timeout=60 * 90, cpu=4)
def download(name: str = "flash-bf16"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download

    repo, rev, fname = GGUF[name]
    p = hf_hub_download(repo, fname, revision=rev, local_dir="/models/clef-gguf")
    models.commit()
    out = {"name": name, "path": p, "gb": round(os.path.getsize(p) / 1e9, 2)}
    print(out)
    return out


def _start(model_path, extra, port=8090):
    cmd = [SERVER_BIN, "-m", model_path, "-ngl", "99", "--port", str(port), "--host", "127.0.0.1", "--metrics", *extra]
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    t0 = time.time()
    for _ in range(1200):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc, round(time.time() - t0, 1)
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


def _run(name, which, args, extra):
    import sys

    sys.path.insert(0, "/root")
    if name == "26b":
        model_path = f"/models/{OURS_26B}"
        # same flags as modal_app.start_server (v1–v9), so --check-lock compares like with like
        server_extra = ["-c", "16384", "-np", "4", "--jinja", "--reasoning-budget", "0", *extra.split()]
    else:
        model_path = f"/models/clef-gguf/{GGUF[name][2]}"
        # Clef: the whole prompt must fit one ubatch; the joint head reads the whole batch, so slots run one at a time
        server_extra = ["-c", "16384", "-ub", "4096", "-b", "4096", "-np", "2", *extra.split()]
    proc, start_s = _start(model_path, server_extra)
    ver = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    try:
        mod = __import__(f"bench.{which}", fromlist=["main"])
        out_dir = f"/results/v11/{which}/{name}"
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        ret = mod.main(base_url="http://127.0.0.1:8090", out_dir=out_dir, args=args, commit=results.commit, model=name, backend="llama")
        smi_after = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        with open(f"/results/v11/_run.jsonl", "a") as f:
            f.write(json.dumps({"which": which, "name": name, "args": args, "server_extra": server_extra, "gpu": smi, "mem_after": smi_after,
                                "server": (ver.stdout + ver.stderr).strip()[-200:], "server_start_s": start_s,
                                "bench_s": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        results.commit()
        return json.loads(json.dumps(ret, default=str))
    finally:
        proc.terminate()


@app.function(image=image, gpu="L40S", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_l40s(name: str = "flash-bf16", which: str = "systemone_bench", args: str = "", extra: str = ""):
    return _run(name, which, args, extra)


@app.function(image=image, gpu="H100", volumes={"/models": models, "/results": results}, timeout=60 * 60)
def run_h100(name: str = "clef-bf16", which: str = "systemone_bench", args: str = "", extra: str = ""):
    return _run(name, which, args, extra)


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 2)
def run_l4(name: str = "26b", which: str = "accuracy", args: str = "", extra: str = ""):
    return _run(name, which, args, extra)
