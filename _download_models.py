"""
_download_models.py  --  called by setup.bat to fetch model files.

Uses huggingface_hub snapshot_download which:
  - Shows a progress bar for each file
  - Resumes interrupted downloads automatically
  - Does NOT require the hf.exe CLI (which can break on some Python versions)
"""

import sys
from pathlib import Path

REPO = "cahlen/lingbot-world-base-cam-nf4"

# Only download exactly the files this app needs (not README, .gitattributes, etc.)
PATTERNS = [
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "high_noise_model_bnb_nf4/model.safetensors",
    "high_noise_model_bnb_nf4/config.json",
    "high_noise_model_bnb_nf4/quantization_meta.json",
    "low_noise_model_bnb_nf4/model.safetensors",
    "low_noise_model_bnb_nf4/config.json",
    "low_noise_model_bnb_nf4/quantization_meta.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
]

LOCAL_DIR = Path(__file__).parent


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("       Run setup.bat again to fix this.")
        return 1

    print(f"  Repository : {REPO}")
    print(f"  Saving to  : {LOCAL_DIR}")
    print()

    try:
        snapshot_download(
            repo_id=REPO,
            local_dir=str(LOCAL_DIR),
            allow_patterns=PATTERNS,
        )
    except KeyboardInterrupt:
        print()
        print("  Download cancelled.  Run setup.bat again to resume.")
        return 1
    except Exception as exc:
        print()
        print(f"  ERROR: {exc}")
        print()
        print("  This is usually a network problem.")
        print("  Run setup.bat again -- it will pick up where it left off.")
        return 1

    print()
    print("  All model files downloaded successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
