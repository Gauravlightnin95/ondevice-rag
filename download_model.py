import huggingface_hub as hf

models = [
    ("OpenVINO/Qwen3-4B-int4-ov",   "models/qwen3-4b-int4"),
    ("OpenVINO/Qwen3-1.7B-int4-ov", "models/qwen3-1.7b-int4"),
    ("OpenVINO/Qwen3-0.6B-int4-ov", "models/qwen3-0.6b-int4"),
]

for repo, path in models:
    print(f"downloading {repo}")
    try:
        hf.snapshot_download(repo, local_dir=path)
    except Exception as e:
        print(f"  failed: {e}")