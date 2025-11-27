# python3 -m modal run app.py

file_name = "audio.mpeg"

import os
import pathlib
import subprocess
import modal

def modal_download():
    # This function is executed in a Modal container to download models and create symlinks.
    from huggingface_hub import hf_hub_download
    
    MODEL_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
    MODEL_FILENAME = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
    MODEL_CACHE_PATH = f"/cache/{MODEL_FILENAME}"

    if not os.path.exists(MODEL_CACHE_PATH):
        print(f"Downloading {MODEL_FILENAME}...")
        subprocess.run(["wget", "-O", MODEL_CACHE_PATH, MODEL_URL], check=True)
        print("Download complete.")

    SYMLINK_DIR = "/root/comfy/ComfyUI/models/TTS/MELBAND"
    SYMLINK_PATH = f"{SYMLINK_DIR}/denoise_mel_band_roformer_sdr_27.99.ckpt"
    os.makedirs(SYMLINK_DIR, exist_ok=True)
    if not os.path.exists(SYMLINK_PATH):
        os.symlink(MODEL_CACHE_PATH, SYMLINK_PATH)
        print(f"Symlink created at {SYMLINK_PATH}")

    # Download and symlink the UVR model from Hugging Face
    UVR_REPO_ID = "SayanoAI/RVC-Studio"
    UVR_FILENAME = "UVR/UVR-DeEcho-DeReverb.pth"
    UVR_SYMLINK_PATH_STR = "/root/comfy/ComfyUI/models/TTS/UVR/UVR-DeEcho-DeReverb.pth"
    
    print(f"Downloading {UVR_FILENAME} from {UVR_REPO_ID}...")
    downloaded_path = hf_hub_download(repo_id=UVR_REPO_ID, filename=UVR_FILENAME, repo_type="dataset")
    print("Download complete.")

    uvr_symlink_path = pathlib.Path(UVR_SYMLINK_PATH_STR)
    uvr_symlink_path.parent.mkdir(parents=True, exist_ok=True)
    if not uvr_symlink_path.exists():
        os.symlink(downloaded_path, uvr_symlink_path)
        print(f"Symlink created at {uvr_symlink_path}")

vol = modal.Volume.from_name("my-cache", create_if_missing=True)
image = (modal.Image.debian_slim()
    .run_commands("apt update")
    .apt_install("git", "ffmpeg", "libsamplerate0-dev", "portaudio19-dev", "wget")
    .uv_pip_install("comfy-cli", "huggingface_hub")
    .env({"HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --nvidia")
    .run_commands("comfy node install tts_audio_suite && python /root/comfy/ComfyUI/custom_nodes/tts_audio_suite/install.py")
    .run_function(modal_download, volumes={"/cache": vol})
    .add_local_file(f"/Users/firozahmed/Downloads/{file_name}", f"/root/comfy/ComfyUI/input/{file_name}")
)

app = modal.App(name="comfyapp", image=image, volumes={"/cache": vol})
# @app.function(max_containers=1, gpu="T4")
# @modal.concurrent(max_inputs=10)  # required for UI startup process which runs several API calls concurrently
# @modal.web_server(8000, startup_timeout=60*3)
# def ui():
#     subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
@app.cls(scaledown_window=300, gpu="L40S")
@modal.concurrent(max_inputs=5)  # run 5 inputs per container
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        subprocess.run(f"comfy launch --background -- --port 8000", shell=True, check=True)
    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        import json
        workflow = {"1":{"inputs":{"model":"MELBAND/denoise_mel_band_roformer_sdr_27.99.ckpt","use_cache":True,"aggressiveness":10,"format":"flac","audio":["2",0]},"class_type":"VocalRemovalNode","_meta":{"title":"🤐 Noise or Vocal Removal"}},"2":{"inputs":{"audio":file_name,"audioUI":""},"class_type":"LoadAudio","_meta":{"title":"LoadAudio"}},"6":{"inputs":{"model":"UVR/UVR-DeEcho-DeReverb.pth","use_cache":True,"aggressiveness":10,"format":"flac","audio":["1",1]},"class_type":"VocalRemovalNode","_meta":{"title":"🤐 Noise or Vocal Removal"}},"9":{"inputs":{"filename_prefix":"ComfyUI","audioUI":"","audio":["6",1]},"class_type":"SaveAudio","_meta":{"title":"SaveAudio"}}}
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f)

        subprocess.run(f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose", shell=True, check=True)
        output_dir = pathlib.Path("/root/comfy/ComfyUI/output")
        
        output_files = sorted(
            (f for f in output_dir.iterdir() if f.is_file() and f.suffix),
            key=lambda f: f.stat().st_mtime, reverse=True
        )
        if not output_files:
            raise FileNotFoundError(f"No output files with an extension found in {output_dir}. Check your ComfyUI workflow. Contents: {list(p.name for p in output_dir.glob('*'))}")
        latest_file = output_files[0]
        return latest_file.read_bytes(), latest_file.suffix
@app.local_entrypoint()
def main():
    output_bytes, extension = ComfyUI().infer.remote()
    output_file_path = f"/Users/firozahmed/Downloads/{pathlib.Path(file_name).stem}_denoise{extension}"
    with open(output_file_path, "wb") as f:
        f.write(output_bytes)
    print(f"File saved to {output_file_path}")