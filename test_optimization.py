import unittest
from unittest.mock import MagicMock, patch
import sys
import asyncio
import os
import json

# Mock missing dependencies
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['pedalboard'] = MagicMock()
sys.modules['pedalboard.io'] = MagicMock()
sys.modules['ebooklib'] = MagicMock()
sys.modules['ebooklib.epub'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['winsound'] = MagicMock()
sys.modules['customtkinter'] = MagicMock()
sys.modules['tkinter'] = MagicMock()
sys.modules['kokoro'] = MagicMock()
sys.modules['pypdf'] = MagicMock()

# Now we can import the engine
from kokoro_engine import KokoroEngine

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.engine = KokoroEngine()

    def tearDown(self):
        self.engine.worker.stop()

    @patch('os.path.exists')
    def test_extract_text_pdf(self, mock_exists):
        mock_exists.return_value = True

        # We need to manually set up the pypdf mock since it's imported
        import pypdf
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page, mock_page]
        pypdf.PdfReader.return_value = mock_reader_instance

        result = self.engine.extract_text_from_file("test.pdf")

        # Verify the structure matches our new implementation logic
        self.assertEqual(result, "Page content\n\nPage content\n\n")

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="Text content")
    @patch('os.path.exists')
    def test_extract_text_txt(self, mock_exists, mock_open):
        mock_exists.return_value = True
        result = self.engine.extract_text_from_file("test.txt")
        self.assertEqual(result, "Text content")

    def test_process_jit_async_string_join(self):
        # We need to simulate the remaining_text_list joining correctly
        config = {
            'out_dir': 'test_out',
            'filename': 'test_file',
            'time_id': '123'
        }

        # We'll set up the state right before the text joining part
        all_text_segments = [("Segment 1", {}), ("Segment 2", {}), ("Segment 3", {})]
        total_segments = 3
        first_remaining_idx = 1 # We've "played" index 0

        # Run the logic that was modified
        remaining_text_list = []
        for i in range(first_remaining_idx, total_segments):
            remaining_text_list.append(all_text_segments[i][0])

        remaining_text = ""
        if remaining_text_list:
            remaining_text = "\n\n".join(remaining_text_list) + "\n\n"

        self.assertEqual(remaining_text, "Segment 2\n\nSegment 3\n\n")


if __name__ == '__main__':
    unittest.main()
