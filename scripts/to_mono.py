"""
Convert WAV files to mono.

Default input:  data/htdemucs_reorganized/{vocals,accompaniment}
Default output: data/htdemucs_mono/{vocals,accompaniment}

Downmix method: average of left/right using audioop.tomono (0.5, 0.5).
Preserves sample rate and sample width.
"""

import argparse
import shutil
from pathlib import Path
import wave
import contextlib
import audioop
from tqdm import tqdm


def convert_wav_to_mono(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(wave.open(str(src_path), 'rb')) as wf_in:
        n_channels = wf_in.getnchannels()
        sampwidth = wf_in.getsampwidth()
        framerate = wf_in.getframerate()
        n_frames = wf_in.getnframes()
        comptype = wf_in.getcomptype()
        compname = wf_in.getcompname()

        if n_channels == 1:
            # Already mono; just copy
            shutil.copy2(src_path, dst_path)
            return

        frames = wf_in.readframes(n_frames)
        # Average L/R into mono
        mono_frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)

    with contextlib.closing(wave.open(str(dst_path), 'wb')) as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(sampwidth)
        wf_out.setframerate(framerate)
        wf_out.setcomptype(comptype, compname)
        wf_out.writeframes(mono_frames)


def process_directory(src_dir: Path, dst_dir: Path) -> None:
    wavs = sorted(src_dir.glob('*.wav'))
    for src in tqdm(wavs, desc=f"{src_dir.name}"):
        dst = dst_dir / src.name
        convert_wav_to_mono(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Convert WAV files to mono (average L/R)")
    parser.add_argument('--input', type=str, default='data/htdemucs_reorganized', help='Input base directory')
    parser.add_argument('--output', type=str, default='data/htdemucs_mono', help='Output base directory')
    args = parser.parse_args()

    in_base = Path(args.input)
    out_base = Path(args.output)

    vocals_in = in_base / 'vocals'
    accomp_in = in_base / 'no_vocals'
    vocals_out = out_base / 'vocals'
    accomp_out = out_base / 'no_vocals'

    if not vocals_in.exists() or not accomp_in.exists():
        raise SystemExit(f"Input subdirectories not found: {vocals_in} / {accomp_in}")

    process_directory(vocals_in, vocals_out)
    process_directory(accomp_in, accomp_out)


if __name__ == '__main__':
    main()


