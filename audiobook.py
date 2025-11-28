import modal

file_name = "input.txt"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-base-ubuntu22.04", 
        add_python="3.11"
    )
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/hf",
        "TORCH_HOME": "/cache/torch",
        "XDG_CACHE_HOME": "/cache",
        "TTS_CACHE": "/cache/tts",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "TESSDATA_PREFIX": "/cache/tessdata",
    })
    .apt_install(
        "git", "calibre", "ffmpeg", "espeak-ng", "sox",
        "libmecab-dev", "mecab", "mecab-ipadic",
        "tesseract-ocr", "tesseract-ocr-eng",
        "pkg-config", "libsndfile1", "curl"
    )
    .run_commands(
        "git clone https://github.com/DrewThomasson/ebook2audiobook.git /app"
    )
    .workdir("/app")
    .uv_pip_install("setuptools", "wheel")
    .run_commands(
        "pip install --no-cache-dir -r requirements.txt"
    )
    .run_commands(
        "python -m unidic download"
    )
    # Create required directories
    .run_commands(
        "mkdir -p /app/audiobooks /app/tmp /app/models /app/.cache && "
        "chmod -R 777 /app/audiobooks /app/tmp /app/models /app/.cache"
    )
    .add_local_file(f"/Users/firozahmed/Downloads/{file_name}", f"/input/{file_name}")
)

app = modal.App("ebook2audiobook", image=image)
volume = modal.Volume.from_name("ebook2audiobook-cache", create_if_missing=True)

@app.cls(
    scaledown_window=300, 
    gpu="T4", 
    volumes={"/cache": volume}, 
    timeout=3600
)
class Ebook2Audiobook:
    @modal.enter()
    def setup(self):
        """Ensure cache directories exist on container start."""
        import os
        os.makedirs("/cache/hf", exist_ok=True)
        os.makedirs("/cache/torch", exist_ok=True)
        os.makedirs("/cache/tts", exist_ok=True)
        os.makedirs("/cache/tessdata", exist_ok=True)

    @modal.method()
    def infer(self, input_filename: str):
        import subprocess
        import glob
        import os
        import shutil
        
        # Set up environment
        os.chdir("/app")
        os.environ["TTS_CACHE"] = "/cache/tts"
        os.environ["TMPDIR"] = "/app/.cache"
        
        # Create fresh output directory for this run
        output_dir = "/app/audiobooks"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = f"/input/{input_filename}"
        
        # Run the Python app directly with correct arguments
        cmd = [
            "python", "/app/app.py",
            "--headless",
            "--ebook", input_path,
            "--language", "eng",
            "--output_dir", output_dir,
            "--script_mode", "native"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd="/app",
            env={
                **os.environ,
                "PYTHONPATH": "/app",
            }
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            # Check if there's still output despite error
            print("Process exited with non-zero code, checking for output files anyway...")
        
        # Find the output file - check multiple possible locations and formats
        search_patterns = [
            "/app/audiobooks/**/*.m4b",
            "/app/audiobooks/**/*.mp3",
            "/app/audiobooks/**/*.wav",
            "/app/audiobooks/*.m4b",
            "/app/audiobooks/*.mp3",
            "/app/audiobooks/*.wav",
        ]
        
        output_files = []
        for pattern in search_patterns:
            output_files.extend(glob.glob(pattern, recursive=True))
        
        if not output_files:
            # Debug: List directory contents
            print("\n=== Directory listing for debugging ===")
            for root, dirs, files in os.walk("/app/audiobooks"):
                print(f"Dir: {root}")
                print(f"  Subdirs: {dirs}")
                print(f"  Files: {files}")
            
            # Also check tmp directory
            print("\n=== /app/tmp contents ===")
            if os.path.exists("/app/tmp"):
                for root, dirs, files in os.walk("/app/tmp"):
                    print(f"Dir: {root}")
                    print(f"  Files: {files}")
            
            raise FileNotFoundError(
                f"No output audio file found. "
                f"Return code: {result.returncode}\n"
                f"STDOUT: {result.stdout[-2000:] if result.stdout else 'empty'}\n"
                f"STDERR: {result.stderr[-2000:] if result.stderr else 'empty'}"
            )
        
        # Get the first (or largest) output file
        output_path = max(output_files, key=os.path.getsize)
        print(f"Found output: {output_path} ({os.path.getsize(output_path)} bytes)")
        extension = os.path.splitext(output_path)[1]
        
        with open(output_path, "rb") as f:
            output_bytes = f.read()
        
        return output_bytes, extension


@app.local_entrypoint()
def main():
    from datetime import datetime
    from pathlib import Path
    
    print(f"Starting conversion of {file_name}...")
    output_bytes, extension = Ebook2Audiobook().infer.remote(file_name)
    
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = f"/Users/firozahmed/Downloads/{Path(file_name).stem}_{date_time}{extension}"
    
    with open(output_file_path, "wb") as f:
        f.write(output_bytes)
    
    print(f"File saved to {output_file_path}")
    print(f"File size: {len(output_bytes) / (1024*1024):.2f} MB")