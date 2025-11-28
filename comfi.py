import subprocess
from pathlib import Path
import modal

file_name = "photo.png"
prompt = "change background to a beautiful beach during sunset"

def hf_download():
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    import subprocess
    import urllib.request
    
    models = [
        ("Phr00t/WAN2.2-14B-Rapid-AllInOne",
         "Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.1.safetensors",
         "checkpoints"),
        ("Phr00t/Qwen-Image-Edit-Rapid-AIO",
         "v11/Qwen-Rapid-AIO-NSFW-v11.1.safetensors",
         "checkpoints"),
        ("Comfy-Org/z_image_turbo",
         "split_files/text_encoders/qwen_3_4b.safetensors",
         "text_encoders"),
        ("Comfy-Org/z_image_turbo",
         "split_files/vae/ae.safetensors",
         "vae"),
        ("Comfy-Org/z_image_turbo",
         "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
         "diffusion_models"),
    ]
    
    base_path = Path("/root/comfy/ComfyUI/models")
    
    for repo_id, filename, model_type in models:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        target_dir = base_path / model_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_name = Path(filename).name
        target_path = target_dir / target_name
        subprocess.run(f"ln -sf {downloaded_path} {target_path}", shell=True, check=True)
    
    # VibeVoice-7B model files
    vibevoice_files = [
        ("aoi-ot/VibeVoice-Large", "model-00001-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00002-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00003-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00004-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00005-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00006-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00007-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00008-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00009-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model-00010-of-00010.safetensors"),
        ("aoi-ot/VibeVoice-Large", "model.safetensors.index.json"),
        ("aoi-ot/VibeVoice-Large", "config.json"),
        ("aoi-ot/VibeVoice-Large", "preprocessor_config.json"),
        # Tokenizer from Qwen2.5-7B (required for VibeVoice-7B)
        ("Qwen/Qwen2.5-7B", "tokenizer.json"),
    ]
    
    vibevoice_dir = base_path / "TTS" / "vibevoice" / "vibevoice-7B"
    vibevoice_dir.mkdir(parents=True, exist_ok=True)
    
    for repo_id, filename in vibevoice_files:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
        target_path = vibevoice_dir / filename
        subprocess.run(f"ln -sf {downloaded_path} {target_path}", shell=True, check=True)
    
    # Direct URL downloads (GitHub releases, HuggingFace datasets, etc.)
    direct_downloads = [
        (
            "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
            "TTS/MELBAND",
            "denoise_mel_band_roformer_sdr_27.99.ckpt"
        ),
        (
            "https://huggingface.co/datasets/SayanoAI/RVC-Studio/resolve/main/UVR/UVR-DeEcho-DeReverb.pth",
            "UVR-DeEcho-DeReverb.pth",
            "TTS/UVR",
            "UVR-DeEcho-DeReverb.pth"
        ),
    ]
    
    cache_dir = Path("/cache/direct_downloads")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for url, cache_name, target_subdir, target_name in direct_downloads:
        cached_path = cache_dir / cache_name
        
        # Download if not already cached
        if not cached_path.exists():
            print(f"Downloading {cache_name}...")
            urllib.request.urlretrieve(url, cached_path)
        
        # Create target directory and symlink
        target_dir = base_path / target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / target_name
        
        subprocess.run(f"ln -sf {cached_path} {target_path}", shell=True, check=True)

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .run_commands("apt update")
    .apt_install("git", "ffmpeg", "libsamplerate0-dev", "portaudio19-dev")
    .uv_pip_install("comfy-cli", "huggingface-hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --nvidia")
    .run_commands("comfy node install ComfyUI-Crystools")
    .run_commands("comfy node install ComfyUI-WanVideoWrapper")
    .run_commands("comfy node install tts_audio_suite && python /root/comfy/ComfyUI/custom_nodes/tts_audio_suite/install.py")
    .run_function(hf_download, volumes={"/cache": vol})
)
app = modal.App(name="comfyapp", image=image, volumes={"/cache": vol})

# @app.function(max_containers=1, gpu="T4")
# @modal.concurrent(max_inputs=10)
# @modal.web_server(8188, startup_timeout=120)
# def ui():
#     subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)

@app.cls(scaledown_window=300, gpu="L40S")
@modal.concurrent(max_inputs=5)  # run 5 inputs per container
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        cmd = f"comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        import json
        import random

        seed = random.randint(1, 1000000)
        workflow_tts = {"110":{"inputs":{"model":"vibevoice-7B","device":"auto","quantize_llm_4bit":True,"attention_mode":"auto","multi_speaker_mode":"Custom Character Switching","cfg_scale":3,"inference_steps":3,"use_sampling":False,"temperature":0.95,"top_p":0.95,"chunk_minutes":0,"max_new_tokens":0},"class_type":"VibeVoiceEngineNode","_meta":{"title":"⚙️ VibeVoice Engine"}},"114":{"inputs":{"text":prompt,"narrator_voice":"voices_examples/Clint_Eastwood CC3 (enhanced2).wav","seed":2584980660,"enable_chunking":True,"max_chars_per_chunk":400,"chunk_combination_method":"auto","silence_between_chunks_ms":100,"enable_audio_cache":True,"batch_size":0,"TTS_engine":["110",0]},"class_type":"UnifiedTTSTextNode","_meta":{"title":"🎤 TTS Text"}},"126":{"inputs":{"filename_prefix":"ComfyUI","audioUI":"","audio":["114",0]},"class_type":"SaveAudio","_meta":{"title":"Save Audio (FLAC)"}}}
        workflow_denoise = {"1":{"inputs":{"model":"MELBAND/denoise_mel_band_roformer_sdr_27.99.ckpt","use_cache":True,"aggressiveness":10,"format":"flac","audio":["2",0]},"class_type":"VocalRemovalNode","_meta":{"title":"🤐 Noise or Vocal Removal"}},"2":{"inputs":{"audio":file_name,"audioUI":""},"class_type":"LoadAudio","_meta":{"title":"LoadAudio"}},"6":{"inputs":{"model":"UVR/UVR-DeEcho-DeReverb.pth","use_cache":True,"aggressiveness":10,"format":"flac","audio":["1",1]},"class_type":"VocalRemovalNode","_meta":{"title":"🤐 Noise or Vocal Removal"}},"9":{"inputs":{"filename_prefix":"ComfyUI","audioUI":"","audio":["6",1]},"class_type":"SaveAudio","_meta":{"title":"SaveAudio"}}}
        workflow_qwen = {"1":{"inputs":{"ckpt_name":"Qwen-Rapid-AIO-NSFW-v11.1.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"2":{"inputs":{"seed":seed,"steps":4,"cfg":1,"sampler_name":"euler_ancestral","scheduler":"beta","denoise":1,"model":["1",0],"positive":["3",0],"negative":["4",0],"latent_image":["11",0]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"3":{"inputs":{"prompt":prompt,"clip":["1",1],"vae":["1",2],"image1":["7",0]},"class_type":"TextEncodeQwenImageEditPlus","_meta":{"title":"TextEncodeQwenImageEditPlus Input Prompt"}},"4":{"inputs":{"prompt":"","clip":["1",1],"vae":["1",2]},"class_type":"TextEncodeQwenImageEditPlus","_meta":{"title":"TextEncodeQwenImageEditPlus Negative (leave blank)"}},"5":{"inputs":{"samples":["2",0],"vae":["1",2]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"7":{"inputs":{"image":file_name},"class_type":"LoadImage","_meta":{"title":"Optional Input Image"}},"10":{"inputs":{"upscale_method":"nearest-exact","megapixels":1,"image":["7",0]},"class_type":"ImageScaleToTotalPixels","_meta":{"title":"Scale Image to Total Pixels"}},"11":{"inputs":{"pixels":["10",0],"vae":["1",2]},"class_type":"VAEEncode","_meta":{"title":"VAE Encode"}},"12":{"inputs":{"filename_prefix":"ComfyUI","images":["5",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}}}
        workflow_wan = {"8":{"inputs":{"seed":seed,"steps":4,"cfg":1,"sampler_name":"euler_ancestral","scheduler":"beta","denoise":1,"model":["32",0],"positive":["28",0],"negative":["28",1],"latent_image":["28",2]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"9":{"inputs":{"text":prompt,"clip":["26",1]},"class_type":"CLIPTextEncode","_meta":{"title":"CLIP Text Encode (Prompt)"}},"10":{"inputs":{"text":"","clip":["26",1]},"class_type":"CLIPTextEncode","_meta":{"title":"Negative Prompt (leave blank cuz 1 CFG)"}},"11":{"inputs":{"samples":["8",0],"vae":["26",2]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"16":{"inputs":{"image":file_name},"class_type":"LoadImage","_meta":{"title":"Start Frame (Optional)"}},"26":{"inputs":{"ckpt_name":"wan2.2-rapid-mega-aio-nsfw-v12.1.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"28":{"inputs":{"width":512,"height":768,"length":["48",0],"batch_size":1,"strength":1,"positive":["9",0],"negative":["10",0],"vae":["26",2],"control_video":["34",0],"control_masks":["34",1]},"class_type":"WanVaceToVideo","_meta":{"title":"T2V=Strength 0, I2V=Strength 1"}},"32":{"inputs":{"shift":8,"model":["26",0]},"class_type":"ModelSamplingSD3","_meta":{"title":"ModelSamplingSD3"}},"34":{"inputs":{"num_frames":["48",0],"empty_frame_level":0.5,"start_index":0,"end_index":-1,"start_image":["16",0]},"class_type":"WanVideoVACEStartToEndFrame","_meta":{"title":"Bypass for T2V, use for I2V"}},"48":{"inputs":{"value":81},"class_type":"PrimitiveInt","_meta":{"title":"Number of Frames"}},"52":{"inputs":{"fps":24,"images":["11",0]},"class_type":"CreateVideo","_meta":{"title":"Create Video"}},"53":{"inputs":{"filename_prefix":"ComfyUI","format":"auto","codec":"h264","video":["52",0]},"class_type":"SaveVideo","_meta":{"title":"Save Video"}}}
        Path(workflow_path).write_text(json.dumps(workflow_tts))
        
        subprocess.run(f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose", shell=True, check=True)
        
        output_dir = "/root/comfy/ComfyUI/output"
        workflow = json.loads(Path(workflow_path).read_text())
        for f in Path(output_dir).iterdir():
            if f.name.startswith("ComfyUI"):
                return f.read_bytes(), f.suffix

@app.local_entrypoint()
def main():
    from datetime import datetime
    output_bytes, extension = ComfyUI().infer.remote()
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = f"/Users/firozahmed/Downloads/{Path(file_name).stem}_{date_time}{extension}"
    with open(output_file_path, "wb") as f:
        f.write(output_bytes)
    print(f"File saved to {output_file_path}")