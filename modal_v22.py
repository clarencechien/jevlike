"""v22 (docs/handoff-v22-glossary-words.md): glossary / per-word arms on E4B and 26B, Modal L4.

  scripts/modal.sh run modal_v22.py::smoke                 # 10 windows per arm on e4b
  scripts/modal.sh run --detach modal_v22.py::suite        # 4 arms x 2 engines, 8 containers -> /results/v22/<engine>_<arm>
  scripts/modal.sh volume get --force gb10-decide-results v22 results/modal/
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
ARMS = ["glossary", "glossary_mixed", "words", "words_glossary"]

app = modal.App("jevlike-v22")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)
llm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([]).apt_install("libgomp1").pip_install("numpy", "requests")
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
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); return proc, round(time.time() - t0, 1)
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate(); raise RuntimeError("llama-server did not become healthy")


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60)
def judge(engine: str = "e4b", arm: str = "glossary", limit: int = 0, root: str = "v22"):
    import sys
    sys.path.insert(0, "/root")
    from bench import v22_judge
    proc, start_s = _start(engine); out_dir = f"/results/{root}/{engine}_{arm}"; os.makedirs(out_dir, exist_ok=True)
    try:
        ret = v22_judge.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=f"--engine {engine} --arm {arm} --workers 4" + (f" --limit {limit}" if limit else ""))
        json.dump({"model": MODEL_PATH[engine], "server_start_s": start_s, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{out_dir}/_run.json", "w"), indent=1)
        results.commit(); return ret
    finally:
        proc.terminate()


@app.local_entrypoint()
def smoke(limit: int = 10):
    for arm in ARMS:
        print(json.dumps(judge.remote("e4b", arm, limit, "v22-smoke"), ensure_ascii=False))


@app.local_entrypoint()
def suite():
    h = [judge.spawn(e, arm, 0, "v22") for e in ("e4b", "26b") for arm in ARMS]
    for x in h:
        print(json.dumps(x.get(), ensure_ascii=False), flush=True)
