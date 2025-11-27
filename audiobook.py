import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-base-ubuntu22.04", 
        add_python="3.12"
    )
    .uv_pip_install("setuptools", "wheel")
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "/cache/hf",
        "TORCH_HOME": "/cache/torch",
        "XDG_CACHE_HOME": "/cache",
    })
    .apt_install(
        "git", "calibre", "ffmpeg",
        "libmecab-dev", "mecab", "mecab-ipadic"
    )
    .run_commands(
        "git clone https://github.com/DrewThomasson/ebook2audiobook.git /app && "
        "cd /app && pip install -r requirements.txt "
    )
    .run_commands(
        "python -m unidic download"
    )
)

app = modal.App("ebook2audiobook", image=image)

@app.function(max_containers=1, gpu="L40s", volumes={"/cache": modal.Volume.from_name("ebook2audiobook-cache", create_if_missing=True)})
@modal.concurrent(max_inputs=10)
@modal.web_server(7860, startup_timeout=600)
def ui():
    import subprocess
    subprocess.Popen("cd /app && python app.py", shell=True)