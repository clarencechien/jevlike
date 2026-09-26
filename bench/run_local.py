"""Run any bench module against an already-running local server (GB10), without Modal.

Mirrors modal_app.run_bench / modal_sglang.run: imports bench.<which> and calls main(base_url, out_dir, args, ...).
Usage:
  python3 bench/run_local.py --which smoke     --base-url http://127.0.0.1:8080 --out-dir results/gb10/smoke
  python3 bench/run_local.py --which accuracy  --base-url http://127.0.0.1:8080 --out-dir results/gb10/accuracy --args "--control-n 0"
  python3 bench/run_local.py --which v4bench   --backend sglang --base-url http://127.0.0.1:30000 --out-dir results/gb10/v4bench/sglang --args "--only L1,L4,L5 --l4-mode warm"
Records the GPU name, server /props (llama) or /get_model_info (SGLang) and the commit in <out-dir>/_run.json.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    which = a["--which"]; base_url = a.get("--base-url", "http://127.0.0.1:8080")
    out_dir = a.get("--out-dir", f"results/gb10/{which}"); args = a.get("--args", "")
    backend = a.get("--backend", "llama"); model = a.get("--model", "26b")
    os.makedirs(out_dir, exist_ok=True)
    mod = __import__(f"bench.{which}", fromlist=["main"])
    t0 = time.time()
    ret = mod.main(base_url=base_url, out_dir=out_dir, args=args, commit=None, model=model, backend=backend)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    try:
        import requests
        info = requests.get(f"{base_url}/props" if backend == "llama" else f"{base_url}/get_model_info", timeout=10).json()
        info = {k: info.get(k) for k in ("model_path", "build_info", "total_slots", "default_generation_settings")} if backend == "llama" else info
    except Exception as e:  # noqa: BLE001
        info = {"error": repr(e)}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(os.path.join(out_dir, "_run.json"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"which": which, "args": args, "backend": backend, "model": model, "gpu": smi, "server": info, "commit": commit,
                            "bench_s": round(time.time() - t0, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ensure_ascii=False) + "\n")
    print(json.dumps(ret, ensure_ascii=False, default=str)[:2000])


if __name__ == "__main__":
    main()
