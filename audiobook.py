import modal
import subprocess
import os
from pathlib import Path
from datetime import datetime

# Configuration
LOCAL_INPUT_PATH = "/Users/firozahmed/Downloads/input.txt"
LOCAL_OUTPUT_DIR = "/Users/firozahmed/Downloads"
LANGUAGE = "eng"
OUTPUT_FORMAT = "mp3"
TTS_ENGINE = "xtts"

vol = modal.Volume.from_name("ebook2audiobook-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "calibre", "espeak-ng")
    .run_commands("git clone https://github.com/DrewThomasson/ebook2audiobook.git /app")
    .workdir("/app")
    .env({"COQUI_TOS_AGREED": "1"})
    .run_commands("pip install -r requirements.txt", "python -m unidic download")
    .add_local_file(LOCAL_INPUT_PATH, "/app/ebooks/input.txt")
)

app = modal.App("ebook2audiobook", image=image, volumes={"/cache": vol})


@app.function(timeout=7200, gpu="T4", memory=8192)
def run_inference():
    import glob
    import shutil
    
    vol.reload()
    
    # Create cache directory
    os.makedirs("/cache/models", exist_ok=True)
    os.makedirs("/app/output", exist_ok=True)
    
    # Symlink /app/models -> /cache/models
    # This catches all model downloads (TTS, Stanza, HF, etc.)
    models_path = "/app/models"
    if os.path.islink(models_path):
        os.unlink(models_path)
    elif os.path.isdir(models_path):
        # Move any existing models to cache first
        for item in os.listdir(models_path):
            src = os.path.join(models_path, item)
            dst = os.path.join("/cache/models", item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(models_path)
    
    os.symlink("/cache/models", models_path)
    
    cmd = [
        "python", "/app/app.py",
        "--headless",
        "--ebook", "/app/ebooks/input.txt",
        "--language", LANGUAGE,
        "--output_format", OUTPUT_FORMAT,
        "--tts_engine", TTS_ENGINE,
        "--output_dir", "/app/output",
    ]
    
    subprocess.run(cmd, check=True)
    vol.commit()
    
    files = glob.glob(f"/app/output/**/*.{OUTPUT_FORMAT}", recursive=True)
    if files:
        output_file = max(files, key=os.path.getmtime)
        with open(output_file, "rb") as f:
            return f.read(), os.path.basename(output_file)
    
    return None, None


@app.local_entrypoint()
def main():
    print(f"Converting: {LOCAL_INPUT_PATH}")
    
    audio_data, filename = run_inference.remote()
    
    if audio_data:
        stem = Path(LOCAL_INPUT_PATH).stem
        ext = Path(filename).suffix if filename else f".{OUTPUT_FORMAT}"
        output_path = f"{LOCAL_OUTPUT_DIR}/{stem}_{datetime.now():%Y%m%d_%H%M%S}{ext}"
        
        Path(output_path).write_bytes(audio_data)
        print(f"✓ Saved: {output_path} ({len(audio_data) / 1024 / 1024:.1f} MB)")
    else:
        print("✗ Failed: No audio generated")