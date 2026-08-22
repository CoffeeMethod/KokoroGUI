"""Static guard enforcing that caching=True only ever appears in
tests/test_caching.py - see tests/conftest.py's make_config docstring."""
import re
from pathlib import Path


def test_no_stray_caching_true_outside_test_caching_module():
    root = Path(__file__).parent
    allowed = {"test_caching.py", "test_meta_caching_policy.py"}
    offenders = []

    for f in sorted(root.glob("test_*.py")):
        if f.name in allowed:
            continue
        content = f.read_text(encoding="utf-8")
        if re.search(r"caching['\"]?\s*[:=]\s*True", content):
            offenders.append(f.name)

    assert not offenders, f"caching=True found outside test_caching.py: {offenders}"
