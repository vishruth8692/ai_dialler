"""Usage: python scripts/test_stt.py path/to/audio.wav [language_code]
language_code defaults to 'unknown' (auto-detect), e.g. pass 'hi-IN' to force Hindi.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.speech.stt_sarvam import transcribe


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_stt.py path/to/audio.wav [language_code]")
        return
    audio_path = sys.argv[1]
    language_code = sys.argv[2] if len(sys.argv) > 2 else "unknown"

    result = transcribe(audio_path, language_code=language_code)
    print(f"Detected language: {result['language_code']}")
    print(f"Transcript: {result['transcript']}")


if __name__ == "__main__":
    main()
