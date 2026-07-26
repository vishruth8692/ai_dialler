"""Buffers streamed text and releases complete sentence-ish chunks as they become available, so
each chunk can be handed to TTS immediately instead of waiting for the whole reply to finish
generating."""

# ASCII enders plus Devanagari danda/double-danda ("।"/"॥") - Claude sometimes replies in native
# Devanagari script (Hindi/Marathi) rather than the Latin-script code-mixed style the prompt
# encourages, and "।" is that script's sentence terminator, not ".".
_SENTENCE_ENDERS = (".", "!", "?", "।", "॥")
_MAX_CHUNK_CHARS = 160  # bounds worst-case TTS-start latency if the model writes one long run-on


class SentenceChunker:
    def __init__(self):
        self._buf = ""

    def feed(self, text: str, final: bool = False) -> list[str]:
        """Feed a text delta, get back zero or more complete chunks ready for TTS.

        Pass final=True on the last call (with text="" if there's nothing more) to flush any
        trailing partial sentence that will never see a sentence-ending punctuation mark.
        """
        self._buf += text
        chunks = []

        while True:
            cut = self._find_cut_point()
            if cut is None:
                break
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if chunk:
                chunks.append(chunk)

        if final:
            trailing = self._buf.strip()
            self._buf = ""
            if trailing:
                chunks.append(trailing)

        return chunks

    def _find_cut_point(self) -> int | None:
        for i, ch in enumerate(self._buf):
            if ch in _SENTENCE_ENDERS:
                return i + 1

        if len(self._buf) > _MAX_CHUNK_CHARS:
            last_space = self._buf.rfind(" ", 0, _MAX_CHUNK_CHARS)
            return last_space + 1 if last_space != -1 else _MAX_CHUNK_CHARS

        return None
