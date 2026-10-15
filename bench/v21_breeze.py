"""v21 Phase 0: Breeze ASR 25 on the exp5 condition files, same settings as kikemu Xbrz_gpu, plus per-token logprobs (D1).

Runs inside modal_v21.py (L4) or on GB10:
  python3 bench/v21_breeze.py --audio <dir with T1__M0.wav ...> --out <dir> [--limit N]

Primary output is one long-form generate(output_scores=True) pass (greedy, timestamps on, language unspecified, fp16):
transcript, segments and per-segment token logprobs from the same decode. The transformers pipeline call kikemu used
(chunk_length_s=0, return_timestamps=True) is run as well and stored as pipeline_transcript for the G0 comparison.
"""
import argparse
import hashlib
import json
import os
import time

MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
REVISION = "cffe7ccb404d025296a00758d0a33468bec3a9d0"
LOW = -1.0  # a token with logprob below this counts as "low confidence" (D1 third feature)


class Breeze:
    def __init__(self, device="cuda"):
        import torch
        import transformers
        from transformers import AutomaticSpeechRecognitionPipeline, WhisperForConditionalGeneration, WhisperProcessor
        t0 = time.time()
        self.proc = WhisperProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        dt = torch.float16 if device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, dtype=dt).to(device).eval()
        self.asr = AutomaticSpeechRecognitionPipeline(model=self.model, tokenizer=self.proc.tokenizer, feature_extractor=self.proc.feature_extractor,
                                                      chunk_length_s=0, dtype=dt, device=device)
        assert self.asr._preprocess_params.get("chunk_length_s") in (0, None)
        self.device, self.dtype = device, dt
        self.load_s = round(time.time() - t0, 1)
        self.versions = {"transformers": transformers.__version__, "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}

    def transcribe(self, audio):
        """audio: float32 mono 16 kHz. Primary output = one long-form generate(output_scores=True) pass: transcript, segments
        (start / end / text) and each segment's token logprob stats (D1) all from the same greedy decode. The transformers
        pipeline call kikemu used is run too and kept as `pipeline_transcript` (it differs in a few tokens: the probe showed
        the pipeline and a bare generate() pick different words here and there); G0 is judged on both."""
        import torch
        t0 = time.time()
        r = self.asr(audio.copy(), return_timestamps=True)
        el = time.time() - t0
        pipe_segments = [{"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]} for c in r.get("chunks", [])]
        t1 = time.time()
        feats = self.proc(audio, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)
        inputs = {k: v.to(self.device) for k, v in feats.items()}
        inputs["input_features"] = inputs["input_features"].to(self.dtype)
        with torch.no_grad():
            g = self.model.generate(**inputs, return_timestamps=True, return_segments=True, return_dict_in_generate=True, output_scores=True)
        el2 = time.time() - t1
        tb = self.model.generation_config.no_timestamps_token_id + 1  # first timestamp token id
        segments = []; n_check = n_argmax = 0
        for s in g["segments"][0]:
            toks = s["tokens"].tolist() if torch.is_tensor(s["tokens"]) else list(s["tokens"])
            res = s["result"]; scores = res["scores"]; seq = res["sequences"]
            n_seq = int(seq.shape[-1]) if torch.is_tensor(seq) else len(seq)
            off = n_seq - len(scores)  # the chunk's decoder prompt (<|startoftranscript|><|lang|><|transcribe|>) has no score
            i0 = int(s["idxs"][0]) - off
            vals = []
            for i, tok in enumerate(toks):
                j = i0 + i
                if tok >= tb or j < 0 or j >= len(scores):
                    continue  # timestamp token / out of range
                sc = scores[j].reshape(-1).float()
                vals.append(float(torch.log_softmax(sc, -1)[tok])); n_check += 1; n_argmax += int(sc.argmax()) == tok
            segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": self.proc.tokenizer.decode([t for t in toks if t < tb], skip_special_tokens=True),
                             "logprob": {"mean": round(sum(vals) / max(len(vals), 1), 4), "min": round(min(vals), 4) if vals else 0.0,
                                         "n": len(vals), "n_low": sum(v < LOW for v in vals)}})
        transcript = self.proc.tokenizer.decode(g["sequences"][0].tolist(), skip_special_tokens=True).strip()
        return {"transcript": transcript, "segments": segments, "pipeline_transcript": r["text"].strip(), "pipeline_segments": pipe_segments,
                "argmax_agree": round(n_argmax / max(n_check, 1), 4), "n_scored_tokens": n_check,
                "elapsed_sec": round(el2, 1), "pipeline_sec": round(el, 1), "audio_s": round(len(audio) / 16000, 1)}


def run(audio_dir, out_dir, limit=0, device="cuda", manifest=None):
    import librosa
    os.makedirs(out_dir, exist_ok=True)
    stems = [f"{s}__{c}" for s in ("T1", "T2", "T3") for c in ("M0", "M1", "M2", "M3")]
    if limit:
        stems = stems[:limit]
    brz = Breeze(device)
    meta = {"arm": "Xbrz_l4" if device == "cuda" else "Xbrz_cpu", "model": MODEL_ID, "revision": REVISION, "dtype": str(brz.dtype), "device": device,
            "chunk_length_s": 0, "return_timestamps": True, "language_arg": None, "corpus": "exp5", "model_load_sec": brz.load_s, **brz.versions,
            "note": "same model/revision/dtype/flags as kikemu Xbrz_gpu (T4 fp16), decoded with generate(output_scores=True) instead of the pipeline; pipeline output kept alongside"}
    json.dump(meta, open(os.path.join(out_dir, "_meta.json"), "w"), ensure_ascii=False, indent=1)
    out = {}
    for stem in stems:
        p = os.path.join(audio_dir, f"{stem}.wav")
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if manifest and manifest["conditions"][stem]["sha256"] != sha:
            raise RuntimeError(f"{stem}.wav sha256 differs from kikemu audio_manifest.json")
        audio, _ = librosa.load(p, sr=16000, mono=True)
        r = brz.transcribe(audio)
        rec = {"arm": meta["arm"], "file": f"{stem}.wav", "audio_s": r["audio_s"], "transcript": r["transcript"], "segments": r["segments"],
               "pipeline_transcript": r["pipeline_transcript"], "pipeline_segments": r["pipeline_segments"],
               "meta": {**meta, "elapsed_sec": r["elapsed_sec"], "pipeline_sec": r["pipeline_sec"], "argmax_agree": r["argmax_agree"],
                        "n_scored_tokens": r["n_scored_tokens"], "same_as_pipeline": r["transcript"] == r["pipeline_transcript"], "audio_sha256": sha}}
        json.dump(rec, open(os.path.join(out_dir, f"{stem}.json"), "w"), ensure_ascii=False, indent=1)
        out[stem] = {"generate_sec": r["elapsed_sec"], "pipeline_sec": r["pipeline_sec"], "n_seg": len(r["segments"]), "n_seg_pipe": len(r["pipeline_segments"]),
                     "argmax_agree": r["argmax_agree"], "same_as_pipeline": r["transcript"] == r["pipeline_transcript"], "head": r["transcript"][:60]}
        print(stem, out[stem], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda"); ap.add_argument("--manifest")
    a = ap.parse_args()
    run(a.audio, a.out, a.limit, a.device, json.load(open(a.manifest)) if a.manifest else None)
