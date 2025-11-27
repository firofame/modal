import os
from pathlib import Path
import subprocess
import modal

vol = modal.Volume.from_name("my-cache", create_if_missing=True)
image = (modal.Image.debian_slim()
    .run_commands("apt update")
    .apt_install("git", "ffmpeg")
    .uv_pip_install("pyannote.audio", "huggingface_hub")
    .uv_pip_install("internetarchive")
    .env({"HF_HOME": "/cache"})
)
app = modal.App(
    name="diarization", 
    image=image, 
    volumes={"/cache": vol}, 
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("archive-secret")
    ]
)

@app.cls(scaledown_window=300, timeout=60*60, gpu="L4")
class Diarization:
    @modal.enter()
    def load_model(self):
        import torch
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=os.environ["HF_TOKEN"]).to(torch.device("cuda"))
        
    @modal.method()
    def run_diarization(self, audio_path: str):
        import torch
        import torchaudio
        from pyannote.audio import Audio
        
        # Convert to WAV
        wav_path = f"/tmp/{Path(audio_path).stem}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", wav_path], check=True, capture_output=True)

        waveform, sample_rate = torchaudio.load(wav_path)
        output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)

        diarization = output.speaker_diarization
        speakers = diarization.labels()
        first_speaker = speakers[0]
        
        audio = Audio()
        speaker_audio = None
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker == first_speaker:
                segment_audio = audio.crop(wav_path, turn)[0]
                speaker_audio = segment_audio if speaker_audio is None else torch.cat((speaker_audio, segment_audio), dim=1)
        
        os.makedirs("/cache/output", exist_ok=True)
        output_path = f"/cache/output/{Path(audio_path).stem}.wav"
        torchaudio.save(output_path, speaker_audio, sample_rate)
        
        os.remove(wav_path)
        os.remove(audio_path)

@app.local_entrypoint()
def main():
    Diarization().run_diarization.remote()