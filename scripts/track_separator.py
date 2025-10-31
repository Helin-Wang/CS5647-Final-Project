import os
import math
from pathlib import Path
import argparse
import subprocess
import shutil
from tqdm import tqdm
import torch

from pydub import AudioSegment
import soundfile as sf
import numpy as np
from spleeter.separator import Separator

INPUT_DIR = "data/raw"
WAV_DIR = "data/raw_wav"
CLIP_DIR = "data/raw_clips"
SPLEETER_OUTPUT_DIR = "data/separated_spleeter"
DEMUCS_OUTPUT_DIR = "data/separated_demucs"

TARGET_SR = 44100
TARGET_CHANNELS = 2
CLIP_SECONDS = 5
KEEP_TAIL_IF_SEC = 1.0

SPLEETER_MODEL_SPEC = "spleeter:2stems"
DEMUCS_MODEL = "htdemucs"  # default demucs model

def au_to_wav(au_path: Path, wav_path: Path, sr: int = TARGET_SR, channels: int = TARGET_CHANNELS):
    audio = AudioSegment.from_file(au_path)
    audio = audio.set_frame_rate(sr).set_channels(channels).set_sample_width(2)
    audio.export(wav_path, format="wav")

def split_wav_to_clips(wav_path: Path, out_dir: Path,
                       clip_len_sec: int = CLIP_SECONDS, keep_tail_if_sec: float = KEEP_TAIL_IF_SEC):
    audio = AudioSegment.from_wav(wav_path)
    total_ms = len(audio)
    clip_ms = int(clip_len_sec * 1000)

    n_full = total_ms // clip_ms
    tail_ms = total_ms - n_full * clip_ms

    stems = []
    for i in range(n_full):
        start = i * clip_ms
        seg = audio[start: start + clip_ms]
        out_path = out_dir / f"{wav_path.stem}_part{i:04d}.wav"
        seg.export(out_path, format="wav")
        stems.append(out_path)

    if tail_ms >= int(keep_tail_if_sec * 1000):
        start = n_full * clip_ms
        seg = audio[start: start + tail_ms]
        out_path = out_dir / f"{wav_path.stem}_part{n_full:04d}.wav"
        seg.export(out_path, format="wav")
        stems.append(out_path)

    return stems

def separate_with_spleeter(clip_files, out_dir: Path, model_spec: str = SPLEETER_MODEL_SPEC):
    print("Step C: Separate 2 stems (vocals / accompaniment) with Spleeter")
    separator = Separator(model_spec)
    for clip in tqdm(clip_files):
        separator.separate_to_file(str(clip), str(out_dir))


def detect_gpu():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def separate_with_demucs(clip_files, out_dir: Path, model: str = DEMUCS_MODEL, two_stems: str = "vocals", device: str = None):
    print("Step C: Separate 2 stems (vocals / accompaniment) with Demucs")
    if shutil.which("demucs") is None:
        raise RuntimeError("Demucs CLI not found. Please install demucs: pip install demucs")

    # Ensure output directory exists; demucs will create model subdir under this
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect GPU if device not specified
    if device is None:
        device = detect_gpu()
    
    print(f"Using device: {device}")

    for clip in tqdm(clip_files):
        # demucs command: demucs -n htdemucs -d cpu --two-stems=vocals input.wav -o out_dir
        cmd = [
            "demucs",
            "-n", model,
            "-d", device,
            f"--two-stems={two_stems}",
            str(clip),
            "-o", str(out_dir),
        ]
        subprocess.run(cmd, check=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert .au to .wav, split into 5s clips, and separate stems using Spleeter or Demucs.")
    parser.add_argument("--genre", type=str, default="metal", help="Genre subfolder under data/raw (e.g., jazz, metal)")
    parser.add_argument("--method", type=str, choices=["spleeter", "demucs"], default="spleeter", help="Separation backend")
    parser.add_argument("--keep-tail-if-sec", type=float, default=KEEP_TAIL_IF_SEC, help="Keep tail if >= seconds")
    # Spleeter options
    parser.add_argument("--spleeter-model", type=str, default=SPLEETER_MODEL_SPEC, help="Spleeter model spec, e.g., spleeter:2stems")
    # Demucs options
    parser.add_argument("--demucs-model", type=str, default=DEMUCS_MODEL, help="Demucs model name, e.g., htdemucs")
    parser.add_argument("--demucs-device", type=str, default=None, help="Demucs device: cpu or cuda (auto-detect if not specified)")
    parser.add_argument("--demucs-two-stems", type=str, default="vocals", help="Demucs two-stems target (vocals or drums/bass/other if supported)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    genre = args.genre
    au_files = sorted(list(Path(INPUT_DIR).glob(f"{genre}/*.au")))
    print(f"Found {len(au_files)} .au files in {INPUT_DIR}")
    
    print("Step A: From .au to .wav")
    (Path(WAV_DIR) / genre).mkdir(parents=True, exist_ok=True)
    wav_files = []
    for au in tqdm(au_files):
        wav_path = Path(WAV_DIR) / genre / (au.stem + ".wav")
        if au.exists():
            au_to_wav(au, wav_path)
        else:
            print(f"Not found {wav_path}")
        wav_files.append(wav_path)
        
    print("Step B: Split 5s clips")
    (Path(CLIP_DIR) / genre).mkdir(parents=True, exist_ok=True)
    clip_files = []
    for wav in tqdm(wav_files):
        out_subdir = Path(CLIP_DIR) / genre
        parts = split_wav_to_clips(wav, out_subdir, clip_len_sec=CLIP_SECONDS, keep_tail_if_sec=args.keep_tail_if_sec)
        clip_files.extend(parts)
    
    clip_files = sorted(list(Path(CLIP_DIR).glob(f"{genre}/*.wav")))
    print(f"Found {len(clip_files)} clips in {CLIP_DIR}")

    # Step C: separation
    if args.method == "spleeter":
        out_dir = Path(SPLEETER_OUTPUT_DIR) / genre
        separate_with_spleeter(clip_files, out_dir, model_spec=args.spleeter_model)
    else:
        out_dir = Path(DEMUCS_OUTPUT_DIR) / genre
        separate_with_demucs(clip_files, out_dir, model=args.demucs_model, two_stems=args.demucs_two_stems, device=args.demucs_device)


if __name__ == "__main__":
    main()