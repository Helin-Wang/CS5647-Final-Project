import os
import math
from pathlib import Path
from tqdm import tqdm

from pydub import AudioSegment
import soundfile as sf
import numpy as np
from spleeter.separator import Separator
INPUT_DIR = "data/raw"
WAV_DIR = "data/raw_wav"
CLIP_DIR = "data/raw_clips"
SEPARATED_DIR = "data/separated_2stems"

TARGET_SR = 44100        # 统一采样率（Spleeter 常见设置）
TARGET_CHANNELS = 2      # 统一为双声道
CLIP_SECONDS = 5         # 切片时长（秒）
KEEP_TAIL_IF_SEC = 1.0   # 最后不足 5s 的尾部，若 ≥ 1s 则也保留

MODEL_SPEC = "spleeter:2stems"

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

def main():
    genre = "jazz"
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
        break
    
    print("Step B: Split 5s clips")
    (Path(CLIP_DIR) / genre).mkdir(parents=True, exist_ok=True)
    clip_files = []
    for wav in tqdm(wav_files):
        out_subdir = Path(CLIP_DIR) / genre
        parts = split_wav_to_clips(wav, out_subdir)
        clip_files.extend(parts)
        break

    print("Step C: Separate 2 stems（vocals / accompaniment）")
    separator = Separator(MODEL_SPEC)
    for clip in tqdm(clip_files):
        out_dir = Path(SEPARATED_DIR) / clip.stem
        (Path(SEPARATED_DIR) / genre).mkdir(parents=True, exist_ok=True)
        separator.separate_to_file(str(clip), str(SEPARATED_DIR))



if __name__ == "__main__":
    main()