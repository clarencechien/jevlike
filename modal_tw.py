"""tw (docs/handoff-tw-script.md): the Hsiao (2026) Taiwan script-dependence jobs on Gemma 4 E2B/E4B/12B/26B/31B and on Clef GGUF.

Run from the repo root:
  scripts/modal.sh run modal_tw.py::smoke --name 26b               # 12 jobs, all arms, prints one rendered prompt
  scripts/modal.sh run --detach modal_tw.py::run_all               # every model, one container each, in parallel
  scripts/modal.sh run --detach modal_tw.py::det --name 26b         # post-hoc: workers 1 vs 4, cache_prompt on vs off
  scripts/modal.sh run modal_tw.py::d0s --name 26b                  # §6: D0 factory set, Traditional vs Simplified
  scripts/modal.sh volume get gb10-decide-results /tw results/modal/tw
Server: the llama.cpp b11371 build in the models volume (v11), same flags as v12. Only data/ (synthetic + data/tw) is mounted.
"""
import json
import os
import subprocess
import time
import urllib.request

import modal

LLAMA_TAG = "b11371"
BUILD_DIR = f"/models/llama-{LLAMA_TAG}"
SERVER_BIN = f"{BUILD_DIR}/bin/llama-server"
PORT = 8080
GEMMA_FLAGS = ["-c", "16384", "-np", "4", "--jinja", "--reasoning-budget", "0", "--swa-full"]
CLEF_FLAGS = ["-c", "16384", "-ub", "4096", "-b", "4096", "-np", "2"]  # v11: whole prompt in one ubatch for the joint head
MODELS = {  # name -> (path in the models volume, server flags, backend, gpu)
    "e2b": ("gemma-4-E2B-it-Q8_0.gguf", GEMMA_FLAGS, "llama", "L4"),
    "e4b": ("gemma-4-E4B-it-Q8_0.gguf", GEMMA_FLAGS, "llama", "L4"),
    "12b": ("v12/gemma-4-12B-it-Q8_0.gguf", GEMMA_FLAGS, "llama", "L4"),
    "26b": ("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", GEMMA_FLAGS, "llama", "L4"),
    "31b": ("v12/gemma-4-31B-it-Q8_0.gguf", ["-c", "8192", *GEMMA_FLAGS[2:]], "llama", "L40S"),  # v12: 16384 + swa-full OOMs
    "clef": ("clef-gguf/Clef-Q8_0.gguf", CLEF_FLAGS, "systemone", "L40S"),
    "clef-flash": ("clef-gguf/Clef-Flash-Q8_0.gguf", CLEF_FLAGS, "systemone", "L40S"),
}

app = modal.App("gb10-decide-tw")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("libgomp1")
    .pip_install("numpy", "requests", "opencc-python-reimplemented==0.1.7")
    .add_local_dir("decide", remote_path="/root/decide")
    .add_local_dir("bench", remote_path="/root/bench")
    .add_local_dir("data", remote_path="/root/data")
)


def _start(name):
    path, flags, _, _ = MODELS[name]
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    cmd = [SERVER_BIN, "-m", f"/models/{path}", "-ngl", "99", "--port", str(PORT), "--host", "127.0.0.1", "--metrics", *flags]
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd)
    t0 = time.time()
    for _ in range(1500):
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            print(f"server healthy after {time.time() - t0:.1f}s", flush=True)
            return proc, round(time.time() - t0, 1)
        except Exception:  # noqa: BLE001
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy")


def _run(name, args, sub):
    import sys

    sys.path.insert(0, "/root")
    from bench import tw_script

    backend = MODELS[name][2]
    if backend == "systemone":
        args = f"--model-name {name} {args}"
    proc, start_s = _start(name)
    out_dir = f"/results/tw/{sub}/{name}"
    try:
        if backend == "llama":  # one rendered prompt for the record
            from decide.client import read_option_probs
            from decide.prompt import SYSTEM, TemplateRenderer, build_messages
            tr = TemplateRenderer(f"http://127.0.0.1:{PORT}")
            p = tr.render_messages(build_messages("台灣是一個國家。", {"instructions": "這句陳述是否正確？", "options": {"A": "是", "B": "否"}}, SYSTEM))
            print(repr(p), flush=True)
            print(json.dumps(read_option_probs(f"http://127.0.0.1:{PORT}", p, ["A", "B"]), ensure_ascii=False)[:600], flush=True)
        t0 = time.time()
        ret = tw_script.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=args, commit=results.commit, model=name, backend=backend)
        ver = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        with open(f"{out_dir}/_run.json", "a") as f:
            f.write(json.dumps({"name": name, "model": MODELS[name][0], "flags": MODELS[name][1], "args": args, "gpu": smi,
                                "server": (ver.stdout + ver.stderr).strip()[-200:], "server_start_s": start_s, "bench_s": round(time.time() - t0, 1),
                                "ret": ret, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, default=str) + "\n")
        results.commit()
        return ret
    finally:
        proc.terminate()


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_l4(name: str = "26b", args: str = "", sub: str = "main"):
    return _run(name, args, sub)


@app.function(image=image, gpu="L40S", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 3)
def run_l40s(name: str = "31b", args: str = "", sub: str = "main"):
    return _run(name, args, sub)


def _fn(name):
    return run_l40s if MODELS[name][3] == "L40S" else run_l4


@app.local_entrypoint()
def smoke(name: str = "26b"):
    print(json.dumps(_fn(name).remote(name, "--limit 12 --reps 1", "smoke"), ensure_ascii=False, indent=1, default=str))


# post-hoc (not in the pre-registered gates): where does the rep-to-rep drift of the main run come from? 26B prod arm only.
DET = {"det_w1_c1": "--arms prod --reps 2 --workers 1 --cache 1", "det_w4_c0": "--arms prod --reps 2 --workers 4 --cache 0",
       "det_w1_c0": "--arms prod --reps 2 --workers 1 --cache 0"}


@app.function(image=image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60)
def d0s_l4(name: str = "26b"):
    """§6: D0 in Traditional, then the same rows in Simplified (data/tw/D0_s), one container, one server."""
    import sys

    sys.path.insert(0, "/root")
    from bench import accuracy

    proc, _ = _start(name)
    out = {}
    try:
        for sub, extra in (("trad", ""), ("simp", "--data-dir /root/data/tw/D0_s")):
            out_dir = f"/results/tw/d0s/{name}/{sub}"
            os.makedirs(out_dir, exist_ok=True)
            out[sub] = accuracy.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=f"--control-n 0 --workers 4 {extra}".strip(),
                                     commit=results.commit, model=name, backend="llama")
            results.commit()
        return out
    finally:
        proc.terminate()


@app.local_entrypoint()
def d0s(name: str = "26b"):
    print(json.dumps(d0s_l4.remote(name), default=str)[:1500])


@app.local_entrypoint()
def det(name: str = "26b"):
    calls = [(sub, _fn(name).spawn(name, args, sub)) for sub, args in DET.items()]
    for sub, c in calls:
        print(sub, json.dumps(c.get(), default=str)[:600], flush=True)


@app.local_entrypoint()
def run_all(names: str = "e2b,e4b,12b,26b,31b,clef,clef-flash", args: str = ""):
    calls = [(n, _fn(n).spawn(n, args, "main")) for n in names.split(",")]
    for n, c in calls:
        try:
            print(n, json.dumps(c.get(), default=str)[:800], flush=True)
        except Exception as e:  # noqa: BLE001
            print(n, "FAILED", repr(e), flush=True)
