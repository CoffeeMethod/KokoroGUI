import sys
from unittest.mock import MagicMock

# Mocking modules as requested
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pedalboard'] = MagicMock()
sys.modules['pedalboard.io'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['ebooklib'] = MagicMock()
sys.modules['ebooklib.epub'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['kokoro'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['winsound'] = MagicMock()

from kokoro_engine import KokoroEngine

def test_apply_lexicon():
    engine = KokoroEngine()

    text = "Hello world, I like apple."
    lexicon = {"hello": "Goodbye", "apple": "banana"}

    res = engine.apply_lexicon(text, lexicon)
    assert res == "Goodbye world, I like banana."

    # Test caching effect
    assert "hello" in engine._lexicon_cache
    assert "apple" in engine._lexicon_cache

    print("Test passed!")

if __name__ == "__main__":
    test_apply_lexicon()
