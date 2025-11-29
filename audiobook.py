import modal
import subprocess
import os

file_name = "input.txt"

def cache_download():
    import os
    os.makedirs("/cache", exist_ok=True)

vol = modal.Volume.from_name("ebook2audiobook-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("apt update")
    .apt_install("git", "ffmpeg", "calibre")
    .run_commands("git clone https://github.com/DrewThomasson/ebook2audiobook.git /app")
    .run_commands("cd /app && pip install -r requirements.txt")
    .run_commands("python -m unidic download")
    .run_function(cache_download, volumes={"/cache": vol})
    .add_local_file(f"/Users/firozahmed/Downloads/{file_name}", f"/{file_name}")
)

app = modal.App("ebook2audiobook", image=image, volumes={"/cache": vol})
@app.function(gpu="T4",  timeout=3600)
def run_inference():
    os.chdir("/app")
    os.makedirs("/output", exist_ok=True)
    
    # Run the conversion
    subprocess.run([
        "python", "/app/app.py", 
        "--headless", 
        "--ebook", f"/{file_name}", 
        "--language", "eng",
        "--output_format", "mp3", 
        "--output_dir", "/output"
    ], check=True)
    
    # Read and return the generated audio file
    output_files = os.listdir("/output")
    if output_files:
        main_output = f"/output/{output_files[0]}"
        with open(main_output, "rb") as f:
            return f.read()
    return None

@app.local_entrypoint()
def main():
    from datetime import datetime
    from pathlib import Path

    audio_data = run_inference.remote()
    
    if audio_data:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"/Users/firozahmed/Downloads/{Path(file_name).stem}_{date_time}.mp3"
        
        with open(output_file_path, "wb") as f:
            f.write(audio_data)
        
        print(f"Audio file saved to {output_file_path}")
    else:
        print("No audio file was generated")