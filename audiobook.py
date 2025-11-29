import modal
import subprocess
import os
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FILE_NAME = "input.txt"
LOCAL_INPUT_PATH = f"/Users/firozahmed/Downloads/{FILE_NAME}"
LOCAL_OUTPUT_DIR = "/Users/firozahmed/Downloads"

# Conversion settings
LANGUAGE = "eng"
OUTPUT_FORMAT = "mp3"
OUTPUT_CHANNEL = "mono"
TTS_ENGINE = "xtts"

# GPU settings
USE_GPU = True

# XTTS-specific settings
SPEED = 1.0
TEMPERATURE = None
ENABLE_TEXT_SPLITTING = False

# Create persistent volume for caching models
vol = modal.Volume.from_name("ebook2audiobook-cache", create_if_missing=True)

# Volume mount path - use a path that won't exist during build
CACHE_PATH = "/vol/cache"

# Build-time environment variables (minimal - no cache paths)
BUILD_ENV_VARS = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "COQUI_TOS_AGREED": "1",
    "CALIBRE_NO_NATIVE_FILEDIALOGS": "1",
}

# Runtime environment variables (includes cache paths)
RUNTIME_ENV_VARS = {
    # Python encoding
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    
    # App settings
    "COQUI_TOS_AGREED": "1",
    "CALIBRE_NO_NATIVE_FILEDIALOGS": "1",
    "GRADIO_DEBUG": "0",
    "DO_NOT_TRACK": "True",
    
    # Temp directories
    "CALIBRE_TEMP_DIR": "/app/tmp",
    "CALIBRE_CACHE_DIRECTORY": "/app/tmp",
    
    # Model cache directories - point to persistent volume
    "HUGGINGFACE_HUB_CACHE": f"{CACHE_PATH}/models/tts",
    "HF_HOME": f"{CACHE_PATH}/models/tts",
    "HF_DATASETS_CACHE": f"{CACHE_PATH}/models/tts",
    "BARK_CACHE_DIR": f"{CACHE_PATH}/models/tts",
    "TTS_CACHE": f"{CACHE_PATH}/models/tts",
    "TORCH_HOME": f"{CACHE_PATH}/models/tts",
    "TTS_HOME": f"{CACHE_PATH}/models",
    "XDG_CACHE_HOME": f"{CACHE_PATH}/models",
    "TESSDATA_PREFIX": f"{CACHE_PATH}/models/tessdata",
    "STANZA_RESOURCES_DIR": f"{CACHE_PATH}/models/stanza",
    "ARGOS_TRANSLATE_PACKAGE_PATH": f"{CACHE_PATH}/models/argostranslate",
    
    # PyTorch settings
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:32,garbage_collection_threshold:0.6,expandable_segments:True",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_LAUNCH_BLOCKING": "1",
    "CUDA_CACHE_MAXSIZE": "2147483648",
    "SUNO_OFFLOAD_CPU": "False",
    "SUNO_USE_SMALL_MODELS": "False",
}

# Build the container image - use minimal env vars during build
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "calibre", "espeak-ng")
    .run_commands(
        "git clone https://github.com/DrewThomasson/ebook2audiobook.git /app"
    )
    .workdir("/app")
    .env(BUILD_ENV_VARS)  # Only minimal env vars during build
    .run_commands("pip install -r requirements.txt")
    .run_commands("python -m unidic download")
    .add_local_file(LOCAL_INPUT_PATH, f"/app/ebooks/{FILE_NAME}")
)

app = modal.App("ebook2audiobook", image=image, volumes={CACHE_PATH: vol})


def setup_cache_dirs():
    """Create cache directories at runtime."""
    cache_dirs = [
        f"{CACHE_PATH}/models",
        f"{CACHE_PATH}/models/tts",
        f"{CACHE_PATH}/models/stanza",
        f"{CACHE_PATH}/models/argostranslate",
        f"{CACHE_PATH}/models/tessdata",
    ]
    
    for dir_path in cache_dirs:
        os.makedirs(dir_path, exist_ok=True)


def set_runtime_env():
    """Set runtime environment variables."""
    for key, value in RUNTIME_ENV_VARS.items():
        os.environ[key] = value


@app.function(
    timeout=7200,
    gpu="T4" if USE_GPU else None,
    memory=8192,
)
def run_inference(
    file_name: str = FILE_NAME,
    language: str = LANGUAGE,
    output_format: str = OUTPUT_FORMAT,
    output_channel: str = OUTPUT_CHANNEL,
    tts_engine: str = TTS_ENGINE,
    use_gpu: bool = USE_GPU,
    speed: float = SPEED,
    temperature: float = None,
    enable_text_splitting: bool = ENABLE_TEXT_SPLITTING,
    voice_file: str = None,
):
    """Convert ebook to audiobook."""
    import glob
    import shutil
    
    # Set runtime environment variables
    set_runtime_env()
    
    # Reload volume to get latest cached models
    vol.reload()
    
    # Setup cache directories
    setup_cache_dirs()
    
    # Ensure working directory
    os.chdir("/app")
    
    # Create required directories
    os.makedirs("/app/tmp", exist_ok=True)
    os.makedirs("/app/audiobooks/cli", exist_ok=True)
    
    # Symlink models directory to cache for persistence
    models_cache = f"{CACHE_PATH}/models"
    models_local = "/app/models"
    
    if os.path.islink(models_local):
        os.unlink(models_local)
    elif os.path.exists(models_local):
        if os.path.isdir(models_local):
            for item in os.listdir(models_local):
                src = os.path.join(models_local, item)
                dst = os.path.join(models_cache, item)
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            shutil.rmtree(models_local)
    
    os.symlink(models_cache, models_local)
    print(f"Linked {models_local} -> {models_cache}")
    
    device = "cuda" if use_gpu else "cpu"
    output_dir = "/app/audiobooks/cli"
    
    cmd = [
        "python", "/app/app.py",
        "--headless",
        "--ebook", f"/app/ebooks/{file_name}",
        "--language", language,
        "--output_format", output_format,
        "--output_channel", output_channel,
        "--tts_engine", tts_engine,
        "--output_dir", output_dir,
    ]
    
    if speed and speed != 1.0:
        cmd.extend(["--speed", str(speed)])
    
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    
    if enable_text_splitting:
        cmd.append("--enable_text_splitting")
    
    if voice_file:
        cmd.extend(["--voice", voice_file])
    
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    subprocess.run(cmd, check=True)
    
    vol.commit()
    print("Models cached to volume")
    
    # Find output files
    output_files = []
    for pattern in [f"{output_dir}/**/*.{output_format}", f"{output_dir}/*.{output_format}"]:
        output_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found output files: {output_files}")
    
    if output_files:
        output_file = max(output_files, key=os.path.getmtime)
        output_filename = os.path.basename(output_file)
        
        print(f"Reading output file: {output_file}")
        
        with open(output_file, "rb") as f:
            audio_data = f.read()
        
        print(f"Audio file size: {len(audio_data) / (1024*1024):.2f} MB")
        return audio_data, output_filename
    
    # Fallback search
    for ext in ["m4b", "m4a", "mp3", "flac", "mp4", "wav", "ogg", "aac"]:
        for pattern in [f"{output_dir}/**/*.{ext}", f"{output_dir}/*.{ext}"]:
            fallback_files = glob.glob(pattern, recursive=True)
            if fallback_files:
                output_file = max(fallback_files, key=os.path.getmtime)
                output_filename = os.path.basename(output_file)
                print(f"Found fallback file: {output_file}")
                with open(output_file, "rb") as f:
                    return f.read(), output_filename
    
    print("No audio files found. Directory contents:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            print(f"  {os.path.join(root, file)}")
    
    return None, None


@app.function()
def list_cached_models():
    """List all cached models in the volume."""
    vol.reload()
    
    cache_info = []
    total_size = 0
    
    for root, dirs, files in os.walk(CACHE_PATH):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                cache_info.append((filepath, size))
                total_size += size
            except OSError:
                pass
    
    return cache_info, total_size


@app.function()
def clear_cache():
    """Clear all cached models."""
    import shutil
    
    vol.reload()
    
    cleared = 0
    for item in os.listdir(CACHE_PATH):
        path = os.path.join(CACHE_PATH, item)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            cleared += 1
        except OSError as e:
            print(f"Error removing {path}: {e}")
    
    vol.commit()
    return cleared


@app.local_entrypoint()
def main():
    """Convert ebook to audiobook."""
    from datetime import datetime
    
    print("=" * 60)
    print("Ebook to Audiobook Conversion")
    print("=" * 60)
    print(f"Input file:     {LOCAL_INPUT_PATH}")
    print(f"Language:       {LANGUAGE}")
    print(f"TTS Engine:     {TTS_ENGINE}")
    print(f"Output format:  {OUTPUT_FORMAT}")
    print(f"Output channel: {OUTPUT_CHANNEL}")
    print(f"GPU enabled:    {USE_GPU}")
    print("=" * 60)
    
    audio_data, output_filename = run_inference.remote(
        file_name=FILE_NAME,
        language=LANGUAGE,
        output_format=OUTPUT_FORMAT,
        output_channel=OUTPUT_CHANNEL,
        tts_engine=TTS_ENGINE,
        use_gpu=USE_GPU,
        speed=SPEED,
        temperature=TEMPERATURE,
        enable_text_splitting=ENABLE_TEXT_SPLITTING,
    )
    
    if audio_data:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(FILE_NAME).stem
        ext = Path(output_filename).suffix if output_filename else f".{OUTPUT_FORMAT}"
        final_filename = f"{stem}_{date_time}{ext}"
        output_file_path = os.path.join(LOCAL_OUTPUT_DIR, final_filename)
        
        with open(output_file_path, "wb") as f:
            f.write(audio_data)
        
        print("=" * 60)
        print("✓ SUCCESS!")
        print(f"✓ Saved to: {output_file_path}")
        print(f"✓ Size: {len(audio_data) / (1024*1024):.2f} MB")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ FAILED: No audio file was generated")
        print("=" * 60)