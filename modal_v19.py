"""v19 (docs/handoff-v19-l0-numeric-coherence.md): L0 rotations, numeric labels, coherence — one judging server on L4.

  scripts/modal.sh run modal_v19.py::smoke                 # all modes, 3 rows per task, catches template/label bugs
  scripts/modal.sh run --detach modal_v19.py::suite        # l0 (all 200 rows) -> numeric N1, N2 (test) -> coherence (test)
  scripts/modal.sh volume get gb10-decide-results v19 results/modal/
Server: b11371, 26B UD-Q4_K_M, -c 16384 -np 4 --jinja --reasoning-budget 0 --swa-full, no MTP (v18). Only synthetic data.
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

BUILD_DIR = "/models/llama-b11371"
SERVER_BIN = f"{BUILD_DIR}/bin/llama-server"
TARGET = "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
PORT = 8080

app = modal.App("jevlike-v19")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([]).apt_install("libgomp1").pip_install("numpy", "requests")
    .add_local_dir("decide", remote_path="/root/decide").add_local_dir("bench", remote_path="/root/bench").add_local_dir("data", remote_path="/root/data")
)
STEPS = [("l0", "--mode l0 --workers 4", "l0"), ("numeric", "--mode numeric --variant N1 --workers 4", "numeric/N1"),
         ("numeric", "--mode numeric --variant N2 --workers 4", "numeric/N2"), ("coherence", "--mode coherence --workers 4", "coherence")]


def _start():
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    cmd = [SERVER_BIN, "-m", TARGET, "-ngl", "99", "-c", "16384", "-np", "4", "--port", str(PORT), "--host", "127.0.0.1",
           "--jinja", "--reasoning-budget", "0", "--swa-full", "--metrics"]
    log = open("/tmp/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT); t0 = time.time()
    for _ in range(1500):
        if proc.poll() is not None:
            print(open("/tmp/server.log").read()[-4000:], flush=True)
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate(); raise RuntimeError("llama-server did not become healthy")


def _run(steps, root, extra=""):
    import sys
    sys.path.insert(0, "/root")
    from bench import v19
    proc = _start(); out = {}
    try:
        ver = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
        for _, args, sub in steps:
            out_dir = f"/results/{root}/{sub}"; os.makedirs(out_dir, exist_ok=True); t0 = time.time()
            print(f"=== {sub}: {args} {extra}", flush=True)
            ret = v19.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=f"{args} {extra}".strip(), commit=results.commit)
            json.dump({"args": args, "extra": extra, "server": (ver.stdout + ver.stderr).strip()[-160:], "bench_s": round(time.time() - t0, 1),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{out_dir}/_run.json", "w"), indent=1)
            results.commit(); out[sub] = ret
        return out
    finally:
        proc.terminate()


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 40)
def smoke(limit: int = 3):
    return _run(STEPS, "v19-smoke", f"--limit {limit} --resume 0")


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def suite(skip: str = ""):
    return _run([s for s in STEPS if s[2] not in skip.split(",")], "v19")
