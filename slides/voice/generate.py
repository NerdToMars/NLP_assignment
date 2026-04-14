"""Generate per-slide and full-talk voice tracks with VoxCPM2.

Usage:
    python generate.py [--device cuda:3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from narration import NARRATION  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--reference", default=str(THIS_DIR / "reference.wav"))
    parser.add_argument("--output-dir", default=str(THIS_DIR / "out"))
    parser.add_argument("--silence-ms", type=int, default=600)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]

    print(f"[voxcpm] loading model on {args.device}", flush=True)
    t0 = time.time()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    print(f"[voxcpm] model loaded in {time.time() - t0:.1f}s", flush=True)

    sample_rate = model.tts_model.sample_rate
    print(f"[voxcpm] output sample_rate = {sample_rate}", flush=True)

    silence = np.zeros(int(sample_rate * args.silence_ms / 1000), dtype=np.float32)
    full_chunks: list[np.ndarray] = []

    for idx, title, text in NARRATION:
        slide_path = out_dir / f"slide_{idx:02d}.wav"
        t_slide = time.time()
        wav = model.generate(
            text=text,
            reference_wav_path=args.reference,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )
        wav = np.asarray(wav, dtype=np.float32)
        sf.write(str(slide_path), wav, sample_rate)
        dur = len(wav) / sample_rate
        print(
            f"[voxcpm] slide {idx:02d} {title:30s} wrote {slide_path.name} "
            f"dur={dur:5.1f}s elapsed={time.time() - t_slide:5.1f}s",
            flush=True,
        )
        full_chunks.append(wav)
        full_chunks.append(silence)

    full_wav = np.concatenate(full_chunks) if full_chunks else np.zeros(0, dtype=np.float32)
    full_path = out_dir / "full_talk.wav"
    sf.write(str(full_path), full_wav, sample_rate)
    total_dur = len(full_wav) / sample_rate
    print(
        f"[voxcpm] wrote {full_path} total dur={total_dur:5.1f}s "
        f"({total_dur / 60:.2f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
