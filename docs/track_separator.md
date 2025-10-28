# Track Separator

## Overview

The `track_separator.py` script is an audio processing tool that:
- Converts `.au` format audio files to `.wav` format
- Splits long audio files into 5-second clips
- Separates each clip into audio stems using the Spleeter model (vocals/accompaniment)

## Usage

```bash
python scripts/track_separator.py
```

## Configuration

You can modify the following parameters in the script:
- `INPUT_DIR`: Input audio directory (default: `data/raw`)
- `CLIP_SECONDS`: Clip duration in seconds (default: 5 seconds)
- `MODEL_SPEC`: Spleeter model specification (default: `spleeter:2stems`)
- Change the `genre` variable to select the music type to process (default: `"metal"`)

## Output Directories

- `data/raw_wav/`: Converted wav files
- `data/raw_clips/`: Split 5-second clips
- `data/separated_2stems/`: Separated audio tracks (vocals/accompaniment)

## Project Structure

```
data/
├── raw/              # Original .au audio files
├── raw_wav/          # Converted .wav files
├── raw_clips/        # Split clips
└── separated_2stems/ # Separated stems (vocals/accompaniment)

scripts/
└── track_separator.py # Audio processing script
```

