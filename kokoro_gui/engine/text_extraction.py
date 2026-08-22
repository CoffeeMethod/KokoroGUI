"""Text extraction from source files (.txt/.pdf/.epub), multi-speaker script
parsing, and long-text splitting into synthesis-sized chunks.

`extract_text_from_file` reads `pypdf`/`ebooklib`/`epub` via `kokoro_engine.pypdf`
/`.ebooklib`/`.epub` (qualified, at call time) rather than importing those names
directly, so that tests can keep monkeypatching them on the `kokoro_engine` module
(e.g. `monkeypatch.setattr(kokoro_engine.pypdf, "PdfReader", FakeReader)`).
"""
import os
import re

from bs4 import BeautifulSoup

import kokoro_engine


class TextExtractionMixin:
    def extract_text_from_file(self, fpath):
        if not os.path.exists(fpath):
            raise FileNotFoundError("File does not exist.")

        text_data = ""
        lower_path = fpath.lower()

        if lower_path.endswith(".pdf"):
            reader = kokoro_engine.pypdf.PdfReader(fpath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_data += extracted + "\n\n"

        elif lower_path.endswith(".epub"):
            book = kokoro_engine.epub.read_epub(fpath, options={'ignore_ncx': True})
            for item in book.get_items():
                if item.get_type() == kokoro_engine.ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text_data += soup.get_text(separator='\n\n') + "\n\n"
        else:
            # Assume text based
            with open(fpath, "r", encoding="utf-8") as f:
                text_data = f.read()

        return text_data

    def parse_multispeaker_text(self, text):
        """
        Parses text for [PresetName]: or [PresetName:FXPresetName]: syntax.
        Returns a list of (speaker_name, fx_name, text_segment)
        """
        # Regex to find [Name]: or [Name:FX]:

        pattern = r"\[([^\]\n]{1,100})\]:\s*"
        matches = list(re.finditer(pattern, text))

        if not matches:
            return [(None, None, text)]

        segments = []
        for i in range(len(matches)):
            raw_name = matches[i].group(1)
            speaker_name = raw_name
            fx_name = None

            if ":" in raw_name:
                parts = raw_name.split(":", 1)
                speaker_name = parts[0].strip()
                fx_name = parts[1].strip()

            start = matches[i].end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            segment_text = text[start:end].strip()
            if segment_text:
                segments.append((speaker_name, fx_name, segment_text))

        return segments

    def smart_split(self, text, chunk_size=3000):
        chunks = []
        current_chunk = []
        current_len = 0
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if len(para) > chunk_size:
                lines = para.split('\n')
                for line in lines:
                    if current_len + len(line) > chunk_size and current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    current_chunk.append(line)
                    current_len += len(line)
            else:
                if current_len + len(para) > chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(para)
                current_len += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return [c for c in chunks if c.strip()]
