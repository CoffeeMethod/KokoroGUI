"""SRT subtitle file generation from a list of generated segments."""


class SrtMixin:
    def generate_srt(self, segments, output_path):
        def format_time(seconds):
            millis = int((seconds - int(seconds)) * 1000)
            seconds = int(seconds)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                current_time = 0.0
                for i, seg in enumerate(segments):
                    start = current_time
                    end = current_time + seg['duration']
                    f.write(f"{i+1}\n")
                    f.write(f"{format_time(start)} --> {format_time(end)}\n")
                    f.write(f"{seg['text'].strip()}\n\n")
                    current_time = end
            return True
        except Exception as e:
            print(f"Failed to generate SRT: {e}")
            return False
