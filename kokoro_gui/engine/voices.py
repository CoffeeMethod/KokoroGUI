"""Custom-voice path resolution and voice-tensor mixing.

Reads `kokoro_engine.CUSTOM_VOICES_DIR` and calls `kokoro_engine.get_thread_pipeline`
qualified, at call time, so tests can keep monkeypatching those names on the
`kokoro_engine` module (e.g. via the `isolated_dirs` fixture).
"""
import asyncio
import os

import torch

import kokoro_engine


class VoiceMixingMixin:
    def resolve_voice_path(self, voice_name):
        """
        Returns the absolute path if it's a custom voice,
        otherwise returns the name as-is (for standard voices).
        """
        # Sanitize voice_name to prevent path traversal
        safe_voice_name = os.path.basename(voice_name)
        # Check if it's a custom voice file
        custom_path = os.path.join(kokoro_engine.CUSTOM_VOICES_DIR, f"{safe_voice_name}.pt")
        if os.path.exists(custom_path):
            return os.path.abspath(custom_path)
        return voice_name

    async def mix_voices(self, v1_name, v2_name, ratio, new_name, op='mix'):
        def _mix():
            try:
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
                    mixed = t1 + t2 * ratio
                elif op == 'subtract':
                    mixed = t1 - t2 * ratio
                elif op == 'multiply':
                    # Lerp between t1 and t1*t2
                    mixed = t1 * (1.0 - ratio) + (t1 * t2) * ratio
                elif op == 'divide':
                    # Lerp between t1 and t1/t2
                    mixed = t1 * (1.0 - ratio) + (t1 / (t2 + 1e-6)) * ratio
                else: # Default: mix (Linear Interpolation)
                    # mixed = v1 * (1 - ratio) + v2 * ratio
                    # ratio is mix of B. If ratio 0, full A. If ratio 1, full B.
                    mixed = t1 * (1.0 - ratio) + t2 * ratio

                # Save
                # Sanitize new_name to prevent path traversal
                safe_new_name = os.path.basename(new_name)
                out_path = os.path.join(kokoro_engine.CUSTOM_VOICES_DIR, f"{safe_new_name}.pt")
                torch.save(mixed, out_path)
                return True, out_path, mixed
            except Exception as e:
                return False, str(e), None

        return await asyncio.to_thread(_mix)
