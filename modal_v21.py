"""v21 (docs/handoff-v21-asr-error-gate.md) Modal fallback: Breeze transcribes, E4B / 26B flag the windows with errors.

  scripts/modal.sh volume put gb10-decide-results <kikemu>/exp5/corpus/conditions v21/audio   # 12 rebuilt wavs (sha256 = manifest)
  scripts/modal.sh run modal_v21.py::breeze_run --limit 1            # Phase 0 smoke, then without --limit  -> /results/v21/breeze
  scripts/modal.sh volume get gb10-decide-results v21/breeze results/modal/v21/ --force
  python3 bench/v21_labels.py --breeze results/modal/v21/breeze --score ; ... --out data/v21/windows.jsonl   # Phase 1 (local)
  scripts/modal.sh run modal_v21.py::judge_run --engine e4b --mode text    # Phase 2: text / swap on e4b and 26b -> /results/v21/<engine>_<mode>
  scripts/modal.sh run modal_v21.py::mmproj_download                       # Phase 3: E4B audio projector (BF16) -> models volume
  scripts/modal.sh run modal_v21.py::audio_smoke                           # Phase 3 smoke (three checks), then judge_run --mode audio
Servers: llama.cpp b11371, -c 16384 -np 4 --jinja --reasoning-budget 0 --swa-full, no MTP. Public podcast audio only.
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
MMPROJ = "/models/v21/mmproj-gemma-4-E4B-it-BF16.gguf"
PORT = 8080
AUDIO = "/results/v21/audio"

app = modal.App("jevlike-v21")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=False)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("jevlike-hf-cache", create_if_missing=True)
llm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([]).apt_install("libgomp1").pip_install("numpy", "requests", "soundfile", "huggingface_hub")
    .add_local_dir("decide", remote_path="/root/decide").add_local_dir("bench", remote_path="/root/bench").add_local_dir("data", remote_path="/root/data")
)
breeze_image = (  # kikemu's pins (Xbrz_gpu ran on transformers 5.15.0 / torch 2.9.0+cu128)
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.9.0", "transformers==5.15.0", "accelerate>=1.0", "librosa>=0.10", "soundfile>=0.12", "numpy<2.3")
    .add_local_dir("bench", remote_path="/root/bench").add_local_dir("data", remote_path="/root/data")
)


@app.function(image=breeze_image, gpu="L4", volumes={"/results": results, "/cache": hf_cache}, timeout=60 * 60)
def breeze(limit: int = 0, root: str = "v21"):
    import sys
    os.environ["HF_HOME"] = "/cache/hf"
    sys.path.insert(0, "/root")
    from bench import v21_breeze
    man = json.load(open("/root/data/v21/kikemu/audio_manifest.json"))
    out = v21_breeze.run(AUDIO, f"/results/{root}/breeze", limit=limit, device="cuda", manifest=man)
    hf_cache.commit(); results.commit()
    return out


@app.local_entrypoint()
def breeze_run(limit: int = 0):
    print(json.dumps(breeze.remote(limit), ensure_ascii=False, indent=1))


def _start(model, mmproj=None):
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD_DIR}/bin:" + os.environ.get("LD_LIBRARY_PATH", "")
    cmd = [SERVER_BIN, "-m", MODEL_PATH[model], "-ngl", "99", "-c", "16384", "-np", "4", "--port", str(PORT), "--host", "127.0.0.1",
           "--jinja", "--reasoning-budget", "0", "--swa-full", "--metrics"]
    if mmproj:
        cmd += ["--mmproj", mmproj]
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


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60 * 2)
def judge(engine: str = "e4b", mode: str = "text", limit: int = 0, root: str = "v21", workers: int = 4):
    import sys
    sys.path.insert(0, "/root")
    from bench import v21_judge
    proc, start_s = _start(engine, MMPROJ if mode == "audio" else None)
    tag = f"{engine}_{mode}"; out_dir = f"/results/{root}/{tag}"; os.makedirs(out_dir, exist_ok=True)
    try:
        ver = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True)
        ret = v21_judge.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir,
                             args=f"--engine {engine} --mode {mode} --workers {workers} --audio-dir {AUDIO}" + (f" --limit {limit}" if limit else ""))
        json.dump({"model": MODEL_PATH[engine], "mmproj": MMPROJ if mode == "audio" else None, "server": (ver.stdout + ver.stderr).strip()[-160:],
                   "server_start_s": start_s, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{out_dir}/_run.json", "w"), indent=1)
        open(f"{out_dir}/_server.log", "w").write(open("/tmp/server.log").read()[-200000:])
        results.commit()
        return ret
    finally:
        proc.terminate()


@app.local_entrypoint()
def judge_run(engine: str = "e4b", mode: str = "text", limit: int = 0):
    print(json.dumps(judge.remote(engine, mode, limit), ensure_ascii=False, default=str)[:3000])


@app.local_entrypoint()
def judge_all(limit: int = 0):
    """Phase 2: text and swap on both engines, four containers in parallel."""
    h = [judge.spawn(e, m, limit) for e in ("e4b", "26b") for m in ("text", "swap")]
    for x in h:
        print(json.dumps(x.get(), ensure_ascii=False, default=str)[:1500], flush=True)


@app.function(image=llm_image, volumes={"/models": models}, timeout=60 * 30)
def mmproj_download(repo: str = "unsloth/gemma-4-E4B-it-GGUF"):
    """Find and download the BF16 audio/vision projector for E4B (PR #21421's author recommends BF16 over F16/Q8_0)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    files = [f for f in list_repo_files(repo) if "mmproj" in f.lower()]
    pick = [f for f in files if "bf16" in f.lower()] or [f for f in files if "f16" in f.lower()]
    if not pick:
        raise RuntimeError(f"no mmproj in {repo}: {files}")
    os.makedirs(os.path.dirname(MMPROJ), exist_ok=True)
    p = hf_hub_download(repo, pick[0], local_dir="/models/v21/_dl")
    os.replace(p, MMPROJ); models.commit()
    return {"repo": repo, "file": pick[0], "all_mmproj": files, "bytes": os.path.getsize(MMPROJ)}


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 40)
def audio_smoke(root: str = "v21"):
    """Phase 3 smoke: (1) /completion with multimodal_data + n_probs returns letter probabilities; (2) transcription of a
    canonical clean English clip and of one Taiwanese code-switched window (Traditional, Latin kept); (3) Q1 on a window
    with a known English error answers B, on a clean window A. Writes /results/v21/audio_smoke.json."""
    import sys
    sys.path.insert(0, "/root")
    from bench import v21_judge
    proc, start_s = _start("e4b", MMPROJ)
    try:
        out = v21_judge.smoke(f"http://127.0.0.1:{PORT}", AUDIO)
        out["server_start_s"] = start_s
        os.makedirs(f"/results/{root}", exist_ok=True)
        json.dump(out, open(f"/results/{root}/audio_smoke.json", "w"), ensure_ascii=False, indent=1)
        open(f"/results/{root}/audio_smoke_server.log", "w").write(open("/tmp/server.log").read()[-100000:])
        results.commit()
        return out
    finally:
        proc.terminate()


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=60 * 60)
def audio_smoke_then_judge(root: str = "v21", workers: int = 2):
    """Phase 3 in one container: smoke, then the full D3 run only if check 1 (letter probabilities readable, option mass >= 0.9) passes."""
    import sys
    sys.path.insert(0, "/root")
    from bench import v21_judge
    proc, start_s = _start("e4b", MMPROJ)
    try:
        out = v21_judge.smoke(f"http://127.0.0.1:{PORT}", AUDIO); out["server_start_s"] = start_s
        os.makedirs(f"/results/{root}", exist_ok=True)
        json.dump(out, open(f"/results/{root}/audio_smoke.json", "w"), ensure_ascii=False, indent=1); results.commit()
        ret = {"smoke": {k: out.get(k) for k in ("check1_probs_readable", "check2_transcription", "check3_q1_direction", "all_pass")}}
        if out.get("check1_probs_readable"):
            out_dir = f"/results/{root}/e4b_audio"; os.makedirs(out_dir, exist_ok=True)
            ret["judge"] = v21_judge.main(base_url=f"http://127.0.0.1:{PORT}", out_dir=out_dir, args=f"--engine e4b --mode audio --workers {workers} --audio-dir {AUDIO}")
            json.dump({"model": MODEL_PATH["e4b"], "mmproj": MMPROJ, "server_start_s": start_s, "route": "chat_logprobs"}, open(f"{out_dir}/_run.json", "w"), indent=1)
            results.commit()
        return ret
    finally:
        proc.terminate()


@app.local_entrypoint()
def audio_run():
    print(json.dumps(audio_smoke_then_judge.remote(), ensure_ascii=False, default=str)[:3000])


@app.local_entrypoint()
def audio_smoke_run():
    print(json.dumps(audio_smoke.remote(), ensure_ascii=False, indent=1, default=str)[:6000])


@app.function(image=llm_image, gpu="L4", volumes={"/models": models, "/results": results}, timeout=900)
def mm_probe(root: str = "v21"):
    """Which /completion prompt form does this build accept for audio? Findings go to /results/v21/probe_mm.json as they arrive."""
    import base64
    import io
    import requests
    import soundfile as sf
    out = {}

    def save():
        json.dump(out, open(f"/results/{root}/probe_mm.json", "w"), ensure_ascii=False, indent=1); results.commit()
    proc, _ = _start("e4b", MMPROJ)
    try:
        p = requests.get(f"http://127.0.0.1:{PORT}/props", timeout=30).json()
        out["props"] = {k: (str(v)[:1200] if k == "chat_template" else v) for k, v in p.items() if k in ("modalities", "model_path", "build_info", "chat_template")}
        save()
        data, sr = sf.read(f"{AUDIO}/T1__M0.wav", dtype="int16"); seg = data[int(10 * sr):int(17 * sr)]
        buf = io.BytesIO(); sf.write(buf, seg, sr, format="WAV", subtype="PCM_16"); b64 = base64.b64encode(buf.getvalue()).decode()
        tests = [("gemma4_turns", "<|turn>user\nTranscribe this audio.\n<__media__><turn|>\n<|turn>model\n"),
                 ("marker_only", "<__media__>"),
                 ("alt_image_marker", "<|turn>user\nTranscribe.\n<__image__><turn|>\n<|turn>model\n")]
        for name, prompt in tests:
            body = {"prompt": {"prompt_string": prompt, "multimodal_data": [b64]}, "n_predict": 16, "temperature": 0, "n_probs": 3, "samplers": []}
            try:
                r = requests.post(f"http://127.0.0.1:{PORT}/completion", json=body, timeout=300)
                out[name] = {"status": r.status_code, "body": r.text[:400]}
            except Exception as e:  # noqa: BLE001
                out[name] = {"error": str(e)[:300]}
            save()
        out["log_tail"] = open("/tmp/server.log").read()[-5000:]; save()
    finally:
        proc.terminate()
    return {k: v for k, v in out.items() if k != "log_tail"}


@app.local_entrypoint()
def mm_probe_run():
    print(json.dumps(mm_probe.remote(), ensure_ascii=False)[:3000])
