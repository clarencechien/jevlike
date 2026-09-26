"""SGLang arm (B1) on Modal: official lmsysorg/sglang:gemma4 image, FP8 weights in a Volume.

  scripts/modal.sh run modal_sglang.py --which download
  scripts/modal.sh run modal_sglang.py --which probe
  scripts/modal.sh run modal_sglang.py --which bench --args "..."
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

FP8_REPO = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic"
FP8_REV = "ed35d7abe5d9"
GPU = os.environ.get("GB10_GPU", "L40S")

app = modal.App("gb10-decide-sglang")
weights = modal.Volume.from_name("gb10-decide-hf", create_if_missing=True)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)

image = (
    modal.Image.from_registry("lmsysorg/sglang:gemma4", add_python=None)
    .entrypoint([])
    .pip_install("huggingface_hub", "numpy", "requests")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir("decide", remote_path="/root/decide")
    .add_local_dir("bench", remote_path="/root/bench")
    .add_local_dir("data", remote_path="/root/data")
)


@app.function(image=image, volumes={"/hf": weights}, timeout=60 * 60 * 2)
def download():
    from huggingface_hub import snapshot_download

    p = snapshot_download(FP8_REPO, revision=FP8_REV, local_dir=f"/hf/{FP8_REPO}")
    weights.commit()
    tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs)
    return {"path": p, "bytes": tot}


def start_server(port=30000, ctx=4096, max_running=32, extra=()):
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", f"/hf/{FP8_REPO}", "--context-length", str(ctx),
           "--max-running-requests", str(max_running), "--mem-fraction-static", "0.8", "--host", "127.0.0.1", "--port", str(port),
           "--log-level", "warning", *extra]
    print("starting:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    for _ in range(1800):
        if proc.poll() is not None:
            raise RuntimeError(f"sglang exited early {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            print(f"sglang healthy after {time.time() - t0:.1f}s", flush=True)
            return proc, time.time() - t0
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("sglang not healthy")


@app.function(image=image, gpu=GPU, volumes={"/hf": weights, "/results": results}, timeout=60 * 60 * 3)
def run(which: str, args: str = "", server_extra: str = ""):
    import sys

    sys.path.insert(0, "/root")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ver = subprocess.run(["python3", "-c", "import sglang; print(sglang.__version__)"], capture_output=True, text=True).stdout.strip()
    proc, t_start = start_server(extra=tuple(server_extra.split()))
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    env = {"gpu": smi, "sglang_version": ver, "server_start_s": round(t_start, 1), "gpu_mem_after_load": mem, "repo": FP8_REPO, "rev": FP8_REV, "server_extra": server_extra}
    print(json.dumps(env), flush=True)
    try:
        out_dir = f"/results/{which}/sglang"
        os.makedirs(out_dir, exist_ok=True)
        json.dump(env, open(f"{out_dir}/_env.json", "w"), indent=1)
        mod = __import__(f"bench.{which}", fromlist=["main"])
        ret = mod.main(base_url="http://127.0.0.1:30000", out_dir=out_dir, args=args, commit=results.commit, model="sglang", backend="sglang")
        results.commit()
        return ret
    finally:
        proc.terminate()


@app.local_entrypoint()
def main(which: str = "probe", args: str = "", server_extra: str = ""):
    if which == "download":
        print(json.dumps(download.remote(), indent=2))
    else:
        print(json.dumps(run.remote(which, args, server_extra), indent=2, ensure_ascii=False, default=str)[:6000])
