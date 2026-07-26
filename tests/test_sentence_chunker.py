import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.sentence_chunker import SentenceChunker


class TestSentenceChunker(unittest.TestCase):
    def test_multi_sentence_single_delta(self):
        chunker = SentenceChunker()
        chunks = chunker.feed("Hello there. How are you? Great.")
        self.assertEqual(chunks, ["Hello there.", "How are you?", "Great."])

    def test_split_across_many_small_deltas(self):
        chunker = SentenceChunker()
        text = "Bike kharab ho gaya. Thoda late ho gaya."
        chunks = []
        for ch in text:
            chunks.extend(chunker.feed(ch))
        self.assertEqual(chunks, ["Bike kharab ho gaya.", "Thoda late ho gaya."])

    def test_max_length_fallback_no_punctuation(self):
        chunker = SentenceChunker()
        text = "word " * 60  # 300 chars, no sentence-ending punctuation at all
        chunks = chunker.feed(text)
        self.assertTrue(chunks, "expected at least one forced chunk from the length fallback")
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 160)
        # cuts should land on word boundaries, not mid-word
        for chunk in chunks:
            self.assertFalse(chunk.startswith(" "))

    def test_trailing_partial_only_flushed_on_final(self):
        chunker = SentenceChunker()
        chunks = chunker.feed("no ending punctuation here")
        self.assertEqual(chunks, [])
        chunks = chunker.feed("", final=True)
        self.assertEqual(chunks, ["no ending punctuation here"])

    def test_final_with_no_pending_text_returns_nothing(self):
        chunker = SentenceChunker()
        chunker.feed("Complete sentence.")
        chunks = chunker.feed("", final=True)
        self.assertEqual(chunks, [])

    def test_devanagari_danda_is_a_sentence_ender(self):
        # Regression test: a real streaming call produced a Devanagari-script reply using "।"
        # instead of "." as the sentence terminator, and the whole reply arrived as one unsplit
        # chunk because "।" wasn't recognized - defeating the point of sentence chunking.
        chunker = SentenceChunker()
        chunks = chunker.feed("bike खराब होना सच में परेशानी की बात है। Chalo, koi baat nahi.")
        self.assertEqual(
            chunks,
            ["bike खराब होना सच में परेशानी की बात है।", "Chalo, koi baat nahi."],
        )


if __name__ == "__main__":
    unittest.main()
