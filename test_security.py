import sys
import unittest
from unittest.mock import MagicMock
import os

# Mock dependencies
sys.modules['soundfile'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['pedalboard'] = MagicMock()
sys.modules['pedalboard.io'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['ebooklib'] = MagicMock()
sys.modules['ebooklib.epub'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['winsound'] = MagicMock()
sys.modules['kokoro'] = MagicMock()

# Now we can import the engine
from kokoro_engine import KokoroEngine

class TestSecurity(unittest.TestCase):
    def setUp(self):
        self.engine = KokoroEngine()

    def test_path_traversal_resolve_voice_path(self):
        payload = "../../../../../etc/passwd"
        path = self.engine.resolve_voice_path(payload)
        self.assertNotIn("etc", path)
        self.assertNotIn("..", path)
        self.assertFalse("passwd" in path and "/" in path)

    def test_path_traversal_load_preset(self):
        payload = "../../../../../etc/passwd"
        # If it doesn't fail parsing or trying to open /etc/passwd, we expect it to look in presets/
        # Just checking if the constructed path in load_preset would have been bad is tricky
        # since load_preset catches exception and returns None.
        # We can mock os.path.exists and open if needed, or rely on the fix in the engine.
        # Let's mock open to see what it tries to open
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{}')) as m:
            with unittest.mock.patch('os.path.exists', return_value=True):
                self.engine.load_preset(payload)
                m.assert_called_once()
                args, kwargs = m.call_args
                self.assertNotIn("..", args[0])
                self.assertNotIn("etc", args[0])

    def test_path_traversal_load_fx_preset(self):
        payload = "../../../../../etc/passwd"
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data='{}')) as m:
            with unittest.mock.patch('os.path.exists', return_value=True):
                self.engine.load_fx_preset(payload)
                m.assert_called_once()
                args, kwargs = m.call_args
                self.assertNotIn("..", args[0])
                self.assertNotIn("etc", args[0])

if __name__ == '__main__':
    unittest.main()
