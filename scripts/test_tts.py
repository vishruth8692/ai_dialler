"""Usage: python scripts/test_tts.py "some text" [language]
Writes output.wav in the current directory and prints the path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.speech.tts_sarvam import synthesize


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_tts.py "some text" [language]')
        return
    text = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "english"

    audio_bytes = synthesize(text, language=language)
    out_path = Path("output.wav")
    out_path.write_bytes(audio_bytes)
    print(f"Wrote {len(audio_bytes)} bytes to {out_path.resolve()}")


if __name__ == "__main__":
    main()
