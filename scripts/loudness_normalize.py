import argparse
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from tqdm import tqdm
import pyloudnorm as pyln


def find_audio_files(input_dir: Path, pattern: str) -> List[Path]:
    return sorted(list(input_dir.rglob(pattern)))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    peak_limit: float,
) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)

    peak = float(np.max(np.abs(normalized))) if normalized.size > 0 else 0.0
    if peak > peak_limit and peak > 0.0:
        normalized = normalized * (peak_limit / peak)
    return normalized


def process_file(
    in_path: Path,
    out_path: Path,
    target_lufs: float,
    peak_limit: float,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        return

    audio, sr = sf.read(in_path, always_2d=False)

    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32)

    normalized = normalize_loudness(audio, sr, target_lufs, peak_limit)

    ensure_parent_dir(out_path)
    sf.write(out_path, normalized, sr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Loudness normalize WAV files to target LUFS with peak limiting. "
            "Preserves directory structure under the output directory."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_wav",
        help="Input directory containing WAV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/normalized_wav",
        help="Output directory for normalized WAV files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="Glob pattern to match files (used with rglob)",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-14.0,
        help="Target integrated loudness in LUFS (e.g., -23.0 broadcast, -14.0 streaming)",
    )
    parser.add_argument(
        "--peak-limit",
        type=float,
        default=0.99,
        help="Hard peak limiter (linear, 0-1) applied after LUFS normalization",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)

    files = find_audio_files(in_dir, args.pattern)
    print(f"Found {len(files)} files under {in_dir} matching {args.pattern}")

    for src in tqdm(files):
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        process_file(
            src,
            dst,
            target_lufs=args.target_lufs,
            peak_limit=args.peak_limit,
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()


