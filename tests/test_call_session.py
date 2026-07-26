import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.call_session import CallSession

_DUMMY_QUESTIONS = [{"question": "How was your day?", "answer": "reference answer", "language": "english"}]


class TestPopPendingUserTurn(unittest.TestCase):
    def _make_session(self) -> CallSession:
        # Pass a non-empty questions list so __post_init__ doesn't hit qa_store/Chroma.
        return CallSession(questions=list(_DUMMY_QUESTIONS))

    def test_pops_dangling_user_entry(self):
        session = self._make_session()
        session.history.append({"role": "assistant", "content": "greeting"})
        session.history.append({"role": "user", "content": "bike kharab ho gaya"})

        result = session.pop_pending_user_turn()

        self.assertEqual(result, "bike kharab ho gaya")
        self.assertEqual(session.history, [{"role": "assistant", "content": "greeting"}])

    def test_returns_none_when_last_entry_is_assistant(self):
        session = self._make_session()
        session.history.append({"role": "user", "content": "hello"})
        session.history.append({"role": "assistant", "content": "hi there"})

        result = session.pop_pending_user_turn()

        self.assertIsNone(result)
        self.assertEqual(len(session.history), 2)  # nothing popped

    def test_returns_none_on_empty_history(self):
        session = self._make_session()
        self.assertIsNone(session.pop_pending_user_turn())

    def test_merge_pattern_matches_intended_usage(self):
        session = self._make_session()
        session.history.append({"role": "user", "content": "bike kharab ho gaya"})

        pending = session.pop_pending_user_turn()
        merged = f"{pending} {'thoda late ho gaya'}".strip() if pending else "thoda late ho gaya"

        self.assertEqual(merged, "bike kharab ho gaya thoda late ho gaya")
        self.assertEqual(session.history, [])


if __name__ == "__main__":
    unittest.main()
