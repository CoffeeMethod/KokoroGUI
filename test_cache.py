import sys
from unittest.mock import MagicMock

# Mock required missing dependencies
for mod in [
    'soundfile', 'torch', 'numpy', 'scipy', 'scipy.signal',
    'pedalboard', 'pedalboard.io', 'pypdf', 'ebooklib',
    'ebooklib.epub', 'bs4', 'winsound', 'kokoro'
]:
    sys.modules[mod] = MagicMock()

# Now we can safely import our code
from kokoro_engine import KokoroEngine
import re

def test_cache():
    engine = KokoroEngine()

    # Cache should be empty initially
    assert len(engine._lexicon_cache) == 0, "Cache should be empty at start"

    lexicon = {
        "hello": "hi",
        "world": "earth"
    }

    text = "Hello World!"

    # First application, should compile and cache
    res1 = engine.apply_lexicon(text, lexicon)
    assert res1 == "hi earth!", f"Expected 'hi earth!', got '{res1}'"
    assert len(engine._lexicon_cache) == 2, "Cache should contain 2 patterns"
    assert "hello" in engine._lexicon_cache
    assert "world" in engine._lexicon_cache

    # Second application, should reuse cache
    text2 = "Say hello to the World"
    res2 = engine.apply_lexicon(text2, lexicon)
    assert res2 == "Say hi to the earth", f"Expected 'Say hi to the earth', got '{res2}'"
    assert len(engine._lexicon_cache) == 2, "Cache size should still be 2"

    print("All tests passed!")

if __name__ == '__main__':
    test_cache()
