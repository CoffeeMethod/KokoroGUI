"""Loading speaker presets (`presets/*.json`) and FX presets (`presets/fx/*.json`)
used by multi-speaker script parsing. Directory names are fixed constants, not
monkeypatched by any test, so no `import kokoro_engine` qualification is needed here.
"""
import json
import os


class PresetsMixin:
    def load_preset(self, name):
        """Loads a preset from the presets directory."""
        # Sanitize name to prevent path traversal
        safe_name = os.path.basename(name)
        preset_path = os.path.join("presets", f"{safe_name}.json")
        if os.path.exists(preset_path):
            try:
                with open(preset_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading preset {name}: {e}")
        return None

    def load_fx_preset(self, name):
        """Loads an FX preset from the presets/fx directory."""
        # Sanitize name to prevent path traversal
        safe_name = os.path.basename(name)
        fx_path = os.path.join("presets", "fx", f"{safe_name}.json")
        if os.path.exists(fx_path):
            try:
                with open(fx_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading FX preset {name}: {e}")
        return None
