import wave
import struct
import math
from pathlib import Path

def generate_alarm_wav(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    duration = 1.5  # 1.5 seconds beep
    n_samples = int(sample_rate * duration)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(n_samples):
            t = i / sample_rate
            freq = 880 if (t % 0.4) < 0.2 else 660
            sample = int(16000 * math.sin(2 * math.pi * freq * t))
            data = struct.pack('<h', sample)
            wav_file.writeframesraw(data)

if __name__ == "__main__":
    assets_dir = Path(__file__).parent / "assets"
    wav_path = assets_dir / "alarm.wav"
    generate_alarm_wav(wav_path)
    print(f"Generated alarm audio file at {wav_path}")
