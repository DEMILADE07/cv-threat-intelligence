from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def iter_avi_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".avi" else []
    return sorted(p for p in path.rglob("*.avi") if p.is_file())


def convert_one(src: Path, output_dir: Path | None, overwrite: bool) -> Path:
    dst = (output_dir / src.with_suffix(".mp4").name) if output_dir else src.with_suffix(".mp4")
    if dst.exists() and not overwrite:
        print(f"skip exists: {dst}")
        return dst

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create mp4 writer: {dst}")

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1

    cap.release()
    writer.release()
    print(f"converted: {src} -> {dst} ({frames} frames)")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AVI files to QuickTime-friendly MP4.")
    parser.add_argument("path", help="AVI file or folder containing AVI files.")
    parser.add_argument("--output-dir", default="", help="Optional folder for converted MP4 files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP4 files.")
    args = parser.parse_args()

    source = Path(args.path)
    output_dir = Path(args.output_dir) if args.output_dir else None
    files = iter_avi_files(source)
    if not files:
        raise SystemExit(f"No .avi files found at: {source}")

    for file in files:
        convert_one(file, output_dir, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
