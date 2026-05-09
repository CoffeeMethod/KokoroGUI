import time
import re

class EngineFast:
    def __init__(self):
        self._lexicon_cache = {}

    def apply_lexicon(self, text, lexicon):
        for src, dest in lexicon.items():
            if not src: continue
            if src not in self._lexicon_cache:
                self._lexicon_cache[src] = re.compile(re.escape(src), re.IGNORECASE)
            text = self._lexicon_cache[src].sub(dest, text)
        return text

class EngineSlow:
    def apply_lexicon(self, text, lexicon):
        for src, dest in lexicon.items():
            if not src: continue
            pattern = re.compile(re.escape(src), re.IGNORECASE)
            text = pattern.sub(dest, text)
        return text

# Generate large lexicon (to defeat re's internal 512 cache)
lexicon = {f"word{i}": f"rep{i}" for i in range(1000)}
text = "This is a word500 and word999 test." * 10

# Slow
engine_slow = EngineSlow()
start = time.time()
for _ in range(100):
    engine_slow.apply_lexicon(text, lexicon)
slow_time = time.time() - start

# Fast
engine_fast = EngineFast()
start = time.time()
for _ in range(100):
    engine_fast.apply_lexicon(text, lexicon)
fast_time = time.time() - start

print(f"Slow: {slow_time:.4f}s")
print(f"Fast: {fast_time:.4f}s")
if slow_time > 0:
    print(f"Improvement: {((slow_time - fast_time) / slow_time) * 100:.2f}%")
