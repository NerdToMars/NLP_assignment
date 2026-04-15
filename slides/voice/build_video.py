"""Build a 1080p slides+audio video from slides/main.pdf and per-slide wavs.

Pipeline:
  1. Rasterize each PDF page to 1920x1080 PNG with pymupdf (letterboxed if needed).
  2. For each slide, use ffmpeg to combine the PNG with its audio into an MP4 segment.
  3. Concatenate all segments into slides/voice/talk.mp4 using the concat demuxer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent
SLIDES_DIR = ROOT.parent
PDF_PATH = SLIDES_DIR / "main.pdf"
AUDIO_DIR = ROOT / "out"
WORK_DIR = ROOT / "video_build"
FINAL_OUT = ROOT / "talk.mp4"

TARGET_W, TARGET_H = 1920, 1080
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def rasterize_pdf() -> list[Path]:
    doc = pymupdf.open(str(PDF_PATH))
    assert doc.page_count == 20, f"expected 20 pages, got {doc.page_count}"
    pngs: list[Path] = []
    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        rect = page.rect
        scale = min(TARGET_W / rect.width, TARGET_H / rect.height)
        matrix = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        raw_path = WORK_DIR / f"raw_{page_idx + 1:02d}.png"
        pix.save(str(raw_path))

        # Letterbox onto exactly 1920x1080 black canvas.
        img = Image.open(raw_path).convert("RGB")
        canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
        x = (TARGET_W - img.width) // 2
        y = (TARGET_H - img.height) // 2
        canvas.paste(img, (x, y))
        final_path = WORK_DIR / f"slide_{page_idx + 1:02d}.png"
        canvas.save(final_path, "PNG", optimize=True)
        raw_path.unlink()
        pngs.append(final_path)
        print(f"[rasterize] slide {page_idx + 1:02d} -> {final_path.name} ({img.width}x{img.height} on {TARGET_W}x{TARGET_H})")
    return pngs


def build_segment(png: Path, wav: Path, out: Path) -> None:
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(png),
        "-i", str(wav),
        "-c:v", "libx264", "-tune", "stillimage",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def concat_segments(segments: list[Path]) -> None:
    concat_list = WORK_DIR / "concat_list.txt"
    concat_list.write_text(
        "\n".join(f"file '{seg.as_posix()}'" for seg in segments) + "\n",
        encoding="utf-8",
    )
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(FINAL_OUT),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    if not PDF_PATH.exists():
        sys.exit(f"missing PDF: {PDF_PATH}")
    if not AUDIO_DIR.exists():
        sys.exit(f"missing audio dir: {AUDIO_DIR}")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    pngs = rasterize_pdf()

    segments: list[Path] = []
    for idx, png in enumerate(pngs, start=1):
        wav = AUDIO_DIR / f"slide_{idx:02d}.wav"
        if not wav.exists():
            sys.exit(f"missing audio for slide {idx}: {wav}")
        seg = WORK_DIR / f"seg_{idx:02d}.mp4"
        build_segment(png, wav, seg)
        print(f"[segment] slide {idx:02d} -> {seg.name}")
        segments.append(seg)

    concat_segments(segments)
    print(f"[concat] wrote {FINAL_OUT}")

    if not args.keep_work:
        shutil.rmtree(WORK_DIR)


if __name__ == "__main__":
    main()
