"""v18 (docs/handoff-v18-mtp.md): Gemma 4 MTP drafter on llama-server, on vs off, same L4, same flags.

  scripts/modal.sh run modal_v18.py::flags                       # does this llama-server build know --spec-type draft-mtp?
  scripts/modal.sh run modal_v18.py::download_mtp                # Q8 drafter -> models volume /models/v18/MTP/
  scripts/modal.sh run --detach modal_v18.py::suite --mtp 0      # accuracy D0 + v4bench L1,L4,L6 (judging server, thinking off)
  scripts/modal.sh run --detach modal_v18.py::suite --mtp 1
  scripts/modal.sh run --detach modal_v18.py::think --mtp 0      # think_tail on the three weak tasks (thinking server)
  scripts/modal.sh run --detach modal_v18.py::think --mtp 1
  scripts/modal.sh volume get gb10-decide-results v18 results/modal/
Server: the b11371 build from jevlike v11 (models volume). Target = our UD-Q4_K_M 26B (the lock file's model). Only synthetic data.
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

LLAMA_TAG = os.environ.get("LLAMA_TAG", "b11371")
BUILD_DIR = f"/models/llama-{LLAMA_TAG}"
SERVER_BIN = f"{BUILD_DIR}/bin/llama-server"
TARGET = "/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
MTP_REPO, MTP_REV = "unsloth/gemma-4-26B-A4B-it-GGUF", "c099eb48e663fd284577b04978a94ffccb261841"  # pinned 2026-10-08 (HF API)
MTP_FILES = {"q8": "MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf", "bf16": "MTP/mtp-gemma-4-26B-A4B-it-BF16.gguf"}
PORT = 8080

app = modal.App("jevlike-v18")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([]).apt_install("libgomp1").pip_install("numpy", "requests")
    .add_local_dir("decide", remote_path="/root/decide").add_local_dir("bench", remote_path="/root/bench").add_local_dir("data", remote_path="/root/data")
)


def _spec(mtp, draft, n_max):
    return ["--model-draft", f"/models/v18/{MTP_FILES[draft]}", "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)] if mtp else []


def _start(mtp, draft="q8", n_max=4, think=False):
    """think=True drops --reasoning-budget 0 (v12 E4); the letter readout via /completion is unaffected either way."""
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    budget = [] if think else ["--reasoning-budget", "0"]
    cmd = [SERVER_BIN, "-m", TARGET, "-ngl", "99", "-c", "16384", "-np", "4", "--port", str(PORT), "--host", "127.0.0.1",
           "--jinja", *budget, "--swa-full", "--metrics", *_spec(mtp, draft, n_max)]
    print("starting:", " ".join(cmd), flush=True)
    log = open("/tmp/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    for _ in range(1500):
        if proc.poll() is not None:
            print(open("/tmp/server.log").read()[-4000:], flush=True)
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc, round(time.time() - t0, 1), " ".join(cmd)
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


def _metrics():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5).read().decode()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e!r}"


def _bench(which, args, arm, out_sub, meta):
    import sys
    sys.path.insert(0, "/root")
    mod = __import__(f"bench.{which}", fromlist=["main"])
    out_dir = f"/results/v18/{out_sub or which}/{arm}"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    ret = mod.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=args, commit=results.commit, model="26b", backend="llama")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ver = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
    open(f"{out_dir}/_metrics.txt", "w").write(_metrics())
    rec = {"which": which, "out_sub": out_sub, "arm": arm, "args": args, "gpu": smi, "server": (ver.stdout + ver.stderr).strip()[-160:],
           "bench_s": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta}
    json.dump(rec, open(f"{out_dir}/_run.json", "w"), indent=1)
    results.commit()
    return json.loads(json.dumps(ret, default=str))


@app.function(image=image, gpu="L4", volumes={"/models": models}, timeout=60 * 10)
def flags():
    """Does this build know the MTP flags? Prints the matching --help lines and the version."""
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    h = subprocess.run([SERVER_BIN, "--help"], capture_output=True, text=True)
    v = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
    lines = [l for l in (h.stdout + h.stderr).splitlines() if "spec" in l.lower() or "draft" in l.lower()]
    out = {"version": (v.stdout + v.stderr).strip()[-200:], "spec_lines": lines[:40], "has_draft_mtp": "draft-mtp" in (h.stdout + h.stderr)}
    print(json.dumps(out, indent=1))
    return out


@app.function(image=cpu_image, volumes={"/models": models}, timeout=60 * 30, cpu=2)
def download_mtp(variant: str = "q8"):
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    p = hf_hub_download(MTP_REPO, MTP_FILES[variant], revision=MTP_REV, local_dir="/models/v18")
    models.commit()
    out = {"path": p, "mb": round(os.path.getsize(p) / 1e6, 1), "s": round(time.time() - t0)}
    print(out)
    return out


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def suite(mtp: int = 0, draft: str = "q8", n_max: int = 4, skip: str = ""):
    """Judging server (thinking off): accuracy D0 test, then v4bench L1/L4/L6. One server start."""
    arm = f"mtp-{draft}-n{n_max}" if mtp else "base"
    proc, start_s, cmd = _start(bool(mtp), draft, n_max, think=False)
    meta = {"server_start_s": start_s, "cmd": cmd, "mtp": bool(mtp)}
    out = {"arm": arm, "server_start_s": start_s}
    try:
        if "accuracy" not in skip:
            out["accuracy"] = _bench("accuracy", "--control-n 0 --workers 4", arm, "accuracy", meta)
        if "v4bench" not in skip:
            out["v4bench"] = _bench("v4bench", "--n 100 --warmup 10 --only L1,L4,L6 --out v18.json", arm, "v4bench", meta)
        return out
    finally:
        open(f"/results/v18/_server_{arm}.log", "w").write(open("/tmp/server.log").read()[-200000:]); results.commit()
        proc.terminate()


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def think(mtp: int = 0, draft: str = "q8", n_max: int = 4, limit: int = 50):
    """Thinking server (no --reasoning-budget 0): think_tail on the three weak tasks, stage B generates up to 512 tokens."""
    arm = f"mtp-{draft}-n{n_max}" if mtp else "base"
    proc, start_s, cmd = _start(bool(mtp), draft, n_max, think=True)
    meta = {"server_start_s": start_s, "cmd": cmd, "mtp": bool(mtp)}
    try:
        args = f"--tasks m_alarm_severity,q_spc_action,p_uph_anomaly --trigger 0.9 --budget 512 --workers 4 --jevbench 0 --limit {limit}"
        return {"arm": arm, "server_start_s": start_s, "think_tail": _bench("think_tail", args, arm, "think", meta)}
    finally:
        open(f"/results/v18/_server_think_{arm}.log", "w").write(open("/tmp/server.log").read()[-200000:]); results.commit()
        proc.terminate()
