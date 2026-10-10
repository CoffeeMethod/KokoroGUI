"""Custom-voice path resolution and voice-tensor mixing (Kokoro's `.pt`
voices).

Reads `runtime.CUSTOM_VOICES_DIR` qualified, at call time, so tests can
monkeypatch it (the `isolated_dirs` fixture). `torch` and `kokoro_engine`
are imported inside `mix_voices`, so importing this module (the engine
package does) doesn't need the `kokoro` package; the mixing itself calls
`kokoro_engine.get_thread_pipeline` by name, which tests patch.
"""
import asyncio
import os

from kokoro_gui.engine import runtime
from kokoro_gui.engines.voice_store import EmbeddingStore

# Kokoro's `.pt` voices: `runtime.CUSTOM_VOICES_DIR` and a project's
# `engines/kokoro/voices/` (Claude/old/PLAN_tbaw_bundle.md section 4).
KOKORO_VOICES = EmbeddingStore("kokoro", ".pt")


class VoiceMixingMixin:
    def resolve_voice_path(self, voice_name, project_dir=None):
        """
        Returns the absolute path if it's a custom voice,
        otherwise returns the name as-is (for standard voices).
        A project-local mix (`<project_dir>/engines/kokoro/voices/<name>.pt`)
        shadows the global `custom_voices/<name>.pt` (grill TB3).
        """
        # Sanitize voice_name to prevent path traversal
        safe_voice_name = os.path.basename(voice_name)
        custom_path = KOKORO_VOICES.find(safe_voice_name, project_dir)
        if custom_path:
            return custom_path
        # Not a custom voice: return the sanitized name (not the raw
        # `voice_name`) so a preset-supplied path/UNC string can't reach
        # `KPipeline`/torch.load as a literal path (see Claude/SECURITY_AUDIT.md).
        # Standard voice names (e.g. "af_bella") have no path separators, so
        # this is a no-op for legitimate names.
        return safe_voice_name

    def _blend(self, v1_name, v2_name, ratio, op='mix'):
        """The blended tensor of two voices (CPU, nothing written). Raises on
        a missing pipeline or voice."""
        import torch

        import kokoro_engine

        # Ensure we have a pipeline to load voices
        # Use 'a' as default for mixing if main pipeline is not ready
        p = self.pipeline
        if not p:
            p = kokoro_engine.get_thread_pipeline('a')
            if not p: raise RuntimeError("No pipeline available for mixing")

        # Resolve inputs (handle custom vs standard)
        v1_arg = self.resolve_voice_path(v1_name)
        v2_arg = self.resolve_voice_path(v2_name)

        # Load tensors
        # KPipeline.load_voice returns a tensor
        t1 = p.load_voice(v1_arg)
        t2 = p.load_voice(v2_arg)

        if t1 is None or t2 is None:
            raise ValueError("Failed to load one of the voices.")

        # Ensure they are on CPU for mixing
        if isinstance(t1, torch.Tensor): t1 = t1.cpu()
        if isinstance(t2, torch.Tensor): t2 = t2.cpu()

        # Check shapes
        if t1.shape != t2.shape:
            # Try to align? Usually kokoro voices are fixed size [510, 1, 256]
            # If different, we might fail or warn.
            print(f"Warning: Voice shapes differ {t1.shape} vs {t2.shape}. Mixing might fail or produce garbage.")

        # Apply operation
        if op == 'add':
            return t1 + t2 * ratio
        if op == 'subtract':
            return t1 - t2 * ratio
        if op == 'multiply':
            # Lerp between t1 and t1*t2
            return t1 * (1.0 - ratio) + (t1 * t2) * ratio
        if op == 'divide':
            # Lerp between t1 and t1/t2
            return t1 * (1.0 - ratio) + (t1 / (t2 + 1e-6)) * ratio
        # Default: mix (Linear Interpolation)
        # mixed = v1 * (1 - ratio) + v2 * ratio
        # ratio is mix of B. If ratio 0, full A. If ratio 1, full B.
        return t1 * (1.0 - ratio) + t2 * ratio

    async def mix_tensor(self, v1_name, v2_name, ratio, op='mix'):
        """Blends two voices in memory: `(True, "", tensor)`, or
        `(False, message, None)`. Writes no file anywhere; the Mixing dock's
        previews use it, `mix_voices` saves what it returns."""
        def _mix():
            try:
                return True, "", self._blend(v1_name, v2_name, ratio, op)
            except Exception as e:
                return False, str(e), None

        return await asyncio.to_thread(_mix)

    async def mix_voices(self, v1_name, v2_name, ratio, new_name, op='mix'):
        def _mix():
            import torch

            try:
                mixed = self._blend(v1_name, v2_name, ratio, op)

                # Save
                # Sanitize new_name to prevent path traversal
                safe_new_name = os.path.basename(new_name)
                out_path = os.path.join(runtime.CUSTOM_VOICES_DIR, f"{safe_new_name}.pt")
                torch.save(mixed, out_path)
                return True, out_path, mixed
            except Exception as e:
                return False, str(e), None

        return await asyncio.to_thread(_mix)
