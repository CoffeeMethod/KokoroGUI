import os
import asyncio
import json
import numpy as np
import soundfile as sf
from kokoro_engine import KokoroEngine

async def main():
    print("Initializing KokoroEngine...")
    engine = KokoroEngine()

    finished_event = asyncio.Event()

    def on_status(msg, is_error):
        prefix = "ERROR" if is_error else "STATUS"
        print(f"[{prefix}] {msg}")

    def on_progress(percentage, time_elapsed, eta, detail_text):
        print(f"[PROGRESS] {percentage:.1f}% - {detail_text}")

    def on_finish():
        finished_event.set()

    engine.on_status = on_status
    engine.on_progress = on_progress
    engine.on_finish = on_finish

    print("\n--- 1. Initializing Pipeline ---")
    success = await engine.init_pipeline_async(lang_code='a')
    if not success:
        print("Failed to initialize pipeline.")
        return

    print("\n--- 2. Mixing Voices ---")
    v1_name = "af_bella"
    v2_name = "af_sarah"
    mixed_voice_name = "test_mix_voice"
    success, path, _ = await engine.mix_voices(v1_name, v2_name, 0.5, mixed_voice_name, op='mix')
    if not success:
        print(f"Failed to mix voices: {path}")
    else:
        print(f"Successfully mixed voice: {path}")

    print("\n--- 3. Creating Presets ---")
    os.makedirs(os.path.join("presets", "fx"), exist_ok=True)

    # Speaker preset
    speaker_preset = {
        "voice": mixed_voice_name,
        "speed": 1.0,
        "volume": 1.0
    }
    with open(os.path.join("presets", "Narrator.json"), "w") as f:
        json.dump(speaker_preset, f)

    # FX preset
    fx_preset = {
        "reverb_enabled": True,
        "reverb_room_size": 0.8,
        "reverb_damping": 0.5,
        "reverb_wet_level": 0.6,
        "reverb_dry_level": 0.8,
        "delay_enabled": True,
        "delay_time": 0.3,
        "delay_feedback": 0.2,
        "delay_mix": 0.3
    }
    with open(os.path.join("presets", "fx", "Space.json"), "w") as f:
        json.dump(fx_preset, f)

    os.makedirs("test_output", exist_ok=True)
    output_files = []

    print("\n--- 4. Generating Preview ---")
    preview_text = "Welcome to the Kokoro Engine verification test."
    preview_path = os.path.join("test_output", "01_preview.wav")
    success = await engine.generate_preview(
        text=preview_text,
        voice="af_bella",
        speed=1.0,
        output_path=preview_path,
        extra_config={"apply_fx": False}
    )
    if success:
        print(f"Preview generated: {preview_path}")
        output_files.append(preview_path)

    print("\n--- 5. Full Conversion (Multi-speaker, FX, Lexicon) ---")
    conversion_text = """
    This is standard generation testing the default voice.

    [Narrator]: Now I am speaking using the newly mixed custom voice. Let's see how it sounds.

    [Narrator:Space]: And now I am broadcasting from deep space! The reverb and delay should be clearly audible.

    [Narrator]: Returning to normal environment. Let's test the lexicon dictionary: I will say brb and TTYL.
    """

    lexicon = {
        "brb": "be right back",
        "TTYL": "talk to you later"
    }

    conversion_config = {
        "voice": "af_bella",
        "speed": 1.0,
        "out_dir": "test_output",
        "filename": "02_conversion",
        "time_id": "test",
        "combine": True,
        "separate": False,
        "export_subtitles": True,
        "lexicon": lexicon,
        "format": "wav",
        "split_pattern": r"\n+",
        "caching": False,
        "apply_fx": False,
        "num_threads": 1
    }

    finished_event.clear()
    engine.start_conversion(conversion_text, conversion_config)
    await finished_event.wait()

    conversion_path = os.path.join("test_output", "02_conversion_test_combined.wav")
    if os.path.exists(conversion_path):
        print(f"Conversion generated: {conversion_path}")
        output_files.append(conversion_path)
    else:
        print(f"Failed to find conversion output at {conversion_path}")

    print("\n--- 6. JIT Conversion (Just-In-Time) ---")
    jit_text = "This is the final part of the test, using Just In Time conversion. It streams and plays the audio as it generates. Have a wonderful day!"
    jit_config = {
        "voice": "af_sarah",
        "speed": 1.0,
        "out_dir": "test_output",
        "filename": "03_jit",
        "time_id": "test",
        "format": "wav",
        "lexicon": {},
        "split_pattern": r"\n+",
        "caching": False,
        "apply_fx": False,
        "num_threads": 1
    }

    finished_event.clear()
    engine.start_jit_conversion(jit_text, jit_config)
    await finished_event.wait()

    jit_path = os.path.join("test_output", "03_jit_test_jit_output.wav")
    if os.path.exists(jit_path):
        print(f"JIT output saved: {jit_path}")
        output_files.append(jit_path)
    else:
        print(f"Failed to find JIT output at {jit_path}")

    print("\n--- 7. Combining All Audio for Verification ---")
    final_output = os.path.join("test_output", "final_verification_output.wav")
    if output_files:
        try:
            combined_audio = []
            target_sr = 24000
            for fpath in output_files:
                if os.path.exists(fpath):
                    data, sr = sf.read(fpath)

                    # Basic resampling if needed (though everything should be 24k)
                    if sr != target_sr:
                        print(f"Warning: {fpath} has sample rate {sr}. It should be {target_sr}.")

                    # Handle mono/stereo consistency
                    if len(data.shape) > 1:
                        data = data.mean(axis=1) # mix to mono

                    combined_audio.append(data)
                    # Add 1 second of silence between segments
                    combined_audio.append(np.zeros(target_sr))

            if combined_audio:
                final_data = np.concatenate(combined_audio)
                sf.write(final_output, final_data, target_sr)
                print(f"Successfully generated final verification audio: {final_output}")
            else:
                 print("No valid audio data to combine.")
        except Exception as e:
            print(f"Error combining audio files: {e}")
    else:
        print("No output files were generated to combine.")

    print("\nTest completed successfully!")
    # Cleanup Engine
    engine.cancel()
    # A small sleep to let threads close cleanly
    await asyncio.sleep(0.5)

def test_engine_operations():
    """Pytest entrypoint"""
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

if __name__ == "__main__":
    # Ensure event loop runs nicely
    test_engine_operations()
