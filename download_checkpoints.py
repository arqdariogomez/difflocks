
import os
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path

# Config
REPO_ID = "arqdariogomez/difflocks-assets-hybrid"

def main():
    print(f"🚀 Downloading assets from {REPO_ID}...")
    token = os.environ.get("HF_TOKEN", None)
    
    try:
        # 1. Download Checkpoints
        print("🔹 Downloading Checkpoints...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="checkpoints/*",
            local_dir=".", 
            token=token
        )
        
        # 2. Download Blender Asset (Restore the file we deleted from Git)
        print("🔹 Downloading Blender Assets...")
        asset_dir = Path("inference/assets")
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download everything in 'assets/' folder of dataset to 'inference/assets/' local
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="assets/*",
            local_dir="inference", # Esto mapeará assets/file -> inference/assets/file
            token=token
        )
        
        print("✅ All assets restored successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        print("💡 Ensure HF_TOKEN is set in Secrets if the dataset is private.")

if __name__ == "__main__":
    main()
