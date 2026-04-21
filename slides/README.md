# G26 Talk — Slides + Narration Runbook

Everything needed to edit the deck and regenerate the narrated video
`slides/voice/talk.mp4`. Works on the NTU GPU box; adapt paths for other
machines.

---

## 1. Layout

```
slides/
├── README.md              <- this file
├── main.tex               <- Beamer source (edit here)
├── main.pdf               <- compiled deck (rebuilt by tectonic)
└── voice/
    ├── reference.m4a      <- original voice sample (any format ffmpeg can read)
    ├── reference.wav      <- 16 kHz mono conversion used by VoxCPM
    ├── narration.py       <- per-slide narration text (edit here)
    ├── generate.py        <- text -> per-slide wavs + full_talk.wav
    ├── build_video.py     <- PDF + wavs -> talk.mp4 (1920x1080)
    ├── .tts_venv/         <- isolated venv for VoxCPM/ffmpeg/pymupdf
    ├── out/               <- slide_01.wav ... slide_18.wav, full_talk.wav
    └── talk.mp4           <- final 1080p narrated video
```

> `.tts_venv` is **separate** from the project's `uv`-managed `.venv`.
> VoxCPM 2.0.2 depends on `datasets<4`, which conflicts with the main project.

---

## 2. Prereqs (one-time)

- Linux + CUDA GPU with at least 12 GB free (VoxCPM uses ~8 GB).
- CUDA driver version 12.4 or newer. Check with `nvidia-smi`.
- `tectonic` on PATH for LaTeX compilation. Verify: `tectonic --version`.
- Internet access on first run so VoxCPM2 weights can be downloaded from
  HuggingFace (~2 GB, cached to `~/.cache/huggingface/hub`).

If `.tts_venv/` already exists, skip this section.

```bash
cd slides/voice
uv venv .tts_venv --python 3.11
VIRTUAL_ENV="$PWD/.tts_venv" uv pip install \
    --index-strategy unsafe-best-match \
    voxcpm imageio-ffmpeg soundfile pymupdf pillow
VIRTUAL_ENV="$PWD/.tts_venv" uv pip install \
    --index-strategy unsafe-best-match \
    --index "pytorch=https://download.pytorch.org/whl/cu124" \
    "torch>=2.5,<2.7" "torchaudio>=2.5,<2.7"
```

Quick smoke test:

```bash
./.tts_venv/bin/python -c "import torch, voxcpm, pymupdf; print(torch.cuda.is_available())"
# expected: True
```

---

## 3. Workflows

Every workflow below ends with rebuilding whichever downstream artifact is
affected. The commands assume the current directory is `slides/` unless
otherwise noted.

### 3.1 Edit slides only (no narration change)

```bash
# edit main.tex in your editor, then:
tectonic --chatter minimal main.tex

# rebuild the final video with existing audio:
cd voice
./.tts_venv/bin/python build_video.py
```

Produces a fresh `voice/talk.mp4`. No GPU needed. ~20 s.

### 3.2 Edit narration text (same voice)

```bash
# edit voice/narration.py
cd voice
./.tts_venv/bin/python generate.py --device cuda:3
./.tts_venv/bin/python build_video.py
./.tts_venv/bin/python build_video.py --jobs 22 --preset faster --crf 19
```

Regenerates every slide's audio (~14 min on a single A6000) then rebuilds
the video. Pass `--device cuda:N` to pick a free GPU;
`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader` shows
free memory per card.

> To regenerate only a subset of slides, comment the others out in
> `narration.py`, run `generate.py`, then uncomment and rerun. `generate.py`
> overwrites existing wavs in `voice/out/` and rebuilds `full_talk.wav`
> from whatever it produced that run -- so either run all 18 at once, or
> regenerate your targeted subset and then run `build_video.py` directly.

### 3.3 Swap the reference voice

```bash
# drop the new sample into voice/reference.m4a (or .wav/.mp3/.flac).
cd voice
FFMPEG=./.tts_venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
$FFMPEG -y -i reference.m4a -ac 1 -ar 16000 reference.wav
./.tts_venv/bin/python generate.py --device cuda:3
./.tts_venv/bin/python build_video.py
```

Ideal reference clips: 15-45 seconds of clean single-speaker speech, no
music, natural reading tone.

### 3.4 Full regeneration (slides + voice + video)

```bash
cd slides
tectonic --chatter minimal main.tex
cd voice
./.tts_venv/bin/python generate.py --device cuda:3
./.tts_venv/bin/python build_video.py
```

---

## 4. Tunables

**`generate.py`**
- `--device cuda:N` — which GPU to use.
- `--cfg-value` (default 2.0) — voice-clone fidelity. Higher = closer to
  reference but can sound stiff. Range 1.5-3.0 is reasonable.
- `--inference-timesteps` (default 10) — more steps = cleaner audio at
  linear cost. 8-15 is the useful range.
- `--silence-ms` (default 600) — gap inserted between slides in
  `full_talk.wav`. Does not affect per-slide wavs or the video.

**`build_video.py`**
- `--keep-work` — leave `voice/video_build/` intact for debugging.
- Edit `TARGET_W`, `TARGET_H` at the top of the file to change resolution
  (e.g. 2560x1440). The 16:9 aspect ratio matches `aspectratio=169` in
  `main.tex`.

**`main.tex`**
- Number of slides must stay in sync with `narration.py`: both currently
  expect exactly 18. `build_video.py` asserts this. If you add or remove
  a slide in `main.tex`, add or remove the matching entry in
  `narration.py` *before* running `generate.py`.

---

## 5. Timing rules of thumb

- VoxCPM2 reads ~155-170 words per minute. 100 words per slide ≈ 38 s of
  audio.
- Target 10-12 min total. The current deck lands at ~15 min including
  silences; trim long slides (3, 8, 12, 13 are the heaviest) if you need
  to drop under 12 min.
- `generate.py` prints `dur=XX.Xs` for every slide. `full_talk.wav` total
  duration is logged at the end.

---

## 6. Troubleshooting

**`torch.cuda.is_available()` is False in `.tts_venv`.**
The default `voxcpm` pull installs `torch` for CUDA 13, which needs a
newer NVIDIA driver. Reinstall with the pinned CUDA 12.4 index shown in
section 2.

**`tectonic` reports overfull vbox on the metropolis title page.**
Cosmetic, ignore. It's a known interaction with the metropolis theme's
`\titlepage`.

**VoxCPM downloads fail or are very slow.**
Set `HF_TOKEN` before running `generate.py` for faster authenticated
downloads. The weights are cached after the first run.

**`build_video.py` errors with `missing audio for slide N`.**
You either removed a `slide_NN.wav` or changed the count in
`narration.py` without regenerating. Run `generate.py` again.

**Video exists but plays silently.**
Check that `ffmpeg -i talk.mp4` lists a `Stream #0:1 ... Audio: aac`
line. If it is missing, the per-slide wavs were probably empty or
corrupt — inspect `voice/out/slide_01.wav` with any audio player, or
re-run `generate.py`.

**You want to distribute the video.**
`talk.mp4` is already muxed with `+faststart`, so it streams fine from
any web server or Slack upload. For YouTube or Drive, no re-encoding
needed.
