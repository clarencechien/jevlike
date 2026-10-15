"""v20 (docs/handoff-v20-speed.md): LLM call / jevlike E4B / jevlike 26B / EG2 on the same items, same L4.

  scripts/modal.sh run modal_v20.py::smoke                       # every arm, 5 items per task
  scripts/modal.sh run --detach modal_v20.py::llama --model 26b   # engines llm + jev on the 26B server  -> /results/v20/{llm_26b,jev_26b}
  scripts/modal.sh run --detach modal_v20.py::llama --model e4b   # engine jev on the E4B server          -> /results/v20/jev_e4b
  scripts/modal.sh run --detach modal_v20.py::eg --device cuda    # EmbeddingGemma 2 on L4                -> /results/v20/eg_cuda
  scripts/modal.sh run --detach modal_v20.py::eg --device cpu     # EmbeddingGemma 2 on 4 vCPU            -> /results/v20/eg_cpu
  scripts/modal.sh volume get gb10-decide-results v20 results/modal/
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

app = modal.App("jevlike-v20")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)
llm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([]).apt_install("libgomp1").pip_install("numpy", "requests")
    .add_local_dir("decide", remote_path="/root/decide").add_local_dir("bench", remote_path="/root/bench").add_local_dir("data", remote_path="/root/data")
)
eg_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "torchaudio", "sentence-transformers>=6.1.0", "transformers>=5.19.0", "numpy", "requests")
    .add_local_dir("decide", remote_path="/root/decide").add_local_dir("bench", remote_path="/root/bench").add_local_dir("data", remote_path="/root/data")
)


def _start(model):
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    cmd = [SERVER_BIN, "-m", MODEL_PATH[model], "-ngl", "99", "-c", "16384", "-np", "4", "--port", str(PORT), "--host", "127.0.0.1",
           "--jinja", "--reasoning-budget", "0", "--swa-full", "--metrics"]
    log = open("/tmp/server.log", "w"); proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT); t0 = time.time()
    for _ in range(1500):
        if proc.poll() is not None:
            print(open("/tmp/server.log").read()[-4000:], flush=True); raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc, round(time.time() - t0, 1)
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate(); raise RuntimeError("llama-server did not become healthy")


def _meta(out_dir, extra):
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        smi = None  # CPU container
    json.dump({"gpu": smi, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}, open(f"{out_dir}/_run.json", "w"), indent=1)


def _llama(model, root, extra_args=""):
    import sys
    sys.path.insert(0, "/root")
    from bench import v20_speed
    proc, start_s = _start(model); out = {}
    try:
        ver = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
        for engine in (("llm", "jev") if model == "26b" else ("jev",)):
            tag = f"{engine}_{model}"; out_dir = f"/results/{root}/{tag}"; os.makedirs(out_dir, exist_ok=True)
            print(f"=== {tag}", flush=True)
            out[tag] = v20_speed.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=f"--engine {engine} --tag {tag} --workers 4 {extra_args}".strip())
            _meta(out_dir, {"model": MODEL_PATH[model], "server": (ver.stdout + ver.stderr).strip()[-160:], "server_start_s": start_s})
            results.commit()
        return out
    finally:
        proc.terminate()


def _eg(device, root, extra_args=""):
    import sys
    sys.path.insert(0, "/root")
    from bench import v20_speed
    tag = f"eg_{device}"; out_dir = f"/results/{root}/{tag}"; os.makedirs(out_dir, exist_ok=True)
    print(f"=== {tag}", flush=True)
    ret = v20_speed.main(base_url="", out_dir=out_dir, args=f"--engine eg --device {device} --tag {tag} {extra_args}".strip())
    cpu = subprocess.run(["bash", "-c", "nproc; grep -m1 'model name' /proc/cpuinfo"], capture_output=True, text=True).stdout.strip()
    _meta(out_dir, {"device": device, "cpu": cpu}); results.commit()
    return ret


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 2)
def llama(model: str = "26b", limit: int = 0, root: str = "v20"):
    return _llama(model, root, f"--limit {limit}" if limit else "")


@app.function(image=eg_image, gpu="L4", volumes={"/results": results}, timeout=60 * 40)
def eg_gpu(limit: int = 0, root: str = "v20"):
    return _eg("cuda", root, f"--limit {limit}" if limit else "")


@app.function(image=eg_image, cpu=4.0, memory=16384, volumes={"/results": results}, timeout=60 * 60)
def eg_cpu(limit: int = 0, root: str = "v20"):
    return _eg("cpu", root, f"--limit {limit}" if limit else "")


@app.local_entrypoint()
def smoke(limit: int = 5):
    """Every arm with a few items per task; results under /results/v20-smoke."""
    print(json.dumps({"llama_26b": llama.remote("26b", limit, "v20-smoke"), "llama_e4b": llama.remote("e4b", limit, "v20-smoke"),
                      "eg_gpu": eg_gpu.remote(limit, "v20-smoke"), "eg_cpu": eg_cpu.remote(limit, "v20-smoke")}, ensure_ascii=False, default=str)[:4000])


@app.local_entrypoint()
def suite():
    """Full run: the four containers in parallel."""
    h = [llama.spawn("26b", 0, "v20"), llama.spawn("e4b", 0, "v20"), eg_gpu.spawn(0, "v20"), eg_cpu.spawn(0, "v20")]
    for x in h:
        print(json.dumps(x.get(), ensure_ascii=False, default=str)[:2000], flush=True)
