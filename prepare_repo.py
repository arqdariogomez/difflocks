import os
import shutil
import zipfile

# --- CONFIGURATION ---
# The name of the file currently on your disk
LOCAL_SOURCE_FILE = "npz_blender_importer.py" 
# The standard name Blender expects inside the zip
BLENDER_STD_NAME = "io_import_difflocks.py"

ADDON_DIR = "blender_addon"
ADDON_ZIP_NAME = "DiffLocks_Blender_Importer.zip"
README_FILE = "README.md"

def create_addon_package():
    print(f"📦 Packaging Blender Add-on...")
    
    # 1. Check if source exists
    if not os.path.exists(LOCAL_SOURCE_FILE):
        # Fallback check
        if os.path.exists(BLENDER_STD_NAME):
            src = BLENDER_STD_NAME
        else:
            print(f"⚠️  WARNING: Could not find '{LOCAL_SOURCE_FILE}'. Skipping Add-on step.")
            return
    else:
        src = LOCAL_SOURCE_FILE

    # 2. Create directory
    if not os.path.exists(ADDON_DIR):
        os.makedirs(ADDON_DIR)

    # 3. Copy and Rename for standard consistency
    dest_py = os.path.join(ADDON_DIR, BLENDER_STD_NAME)
    shutil.copy(src, dest_py)
    print(f"   -> Copied script to: {dest_py}")

    # 4. Create the .zip
    zip_path = os.path.join(ADDON_DIR, ADDON_ZIP_NAME)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(dest_py, arcname=BLENDER_STD_NAME)
    
    print(f"   -> Created ZIP: {zip_path}")

def create_readme():
    print(f"📝 Generating README.md...")
    
    # Safe list approach to avoid SyntaxErrors
    lines = [
        "# DiffLocks Studio (Docker Edition)",
        "",
        "**High-Fidelity 3D Hair Generation from Single Images.**",
        "*Fork optimized for Local usage (Docker) and Google Colab.*",
        "",
        "## 🚀 Features",
        "*   **One-Click Docker Setup:** Run locally on your GPU without messing up your Python environment.",
        "*   **Native 3D Preview:** Visualize results directly in the browser via Plotly.",
        "*   **Blender Integration:** Includes a custom Add-on to import the generated `.npz` hair strands.",
        "*   **Reactive UI:** Real-time console logs and progress tracking.",
        "",
        "---",
        "",
        "## 🛠️ Installation (Local Docker)",
        "",
        "### Prerequisites",
        "*   **Docker Desktop** installed and running.",
        "*   **NVIDIA GPU** (RTX 2060 or higher recommended).",
        "*   **NVIDIA Drivers** updated.",
        "",
        "### 1. Setup",
        "Clone this repository and navigate to the folder:",
        "```bash",
        "git clone https://github.com/arqdariogomez/difflocks.git",
        "cd difflocks",
        "```",
        "",
        "### 2. Prepare Checkpoints",
        "You need to download the model weights from the [Meshcapade website](https://difflocks.is.tue.mpg.de).",
        "Place them in the `checkpoints/` folder following this exact structure:",
        "",
        "```text",
        "checkpoints/",
        "├── difflocks_diffusion/",
        "│   └── scalp_v9_40k_06730000.pth",
        "├── rgb2material/",
        "│   └── rgb2material.pt",
        "└── strand_vae/",
        "    └── strand_codec.pt",
        "```",
        "",
        "### 3. Run",
        "Open a terminal in the folder and run:",
        "",
        "```bash",
        "docker compose up --build",
        "```",
        "",
        "Wait for the installation to finish. Once ready, open your browser at:",
        "**http://localhost:7860**",
        "",
        "---",
        "",
        "## 🔌 Blender Add-on",
        "",
        "To import the generated hair into Blender:",
        "",
        "1.  Go to the `blender_addon/` folder in this repo.",
        "2.  Download `DiffLocks_Blender_Importer.zip`.",
        "3.  In Blender: `Edit > Preferences > Add-ons > Install...` and select the zip.",
        "4.  Enable the add-on.",
        "5.  Import via: `File > Import > DiffLocks Hair (.npz)`.",
        "",
        "---",
        "",
        "## ☁️ Google Colab",
        "You can also run this project in the cloud using the provided Notebook."
    ]
    
    # Write lines joining them with newlines
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print("   -> README.md updated successfully.")

def check_docker_files():
    print(f"🐳 Checking Docker Configuration...")
    required = ["Dockerfile", "docker-compose.yml", "app.py"]
    missing = False
    for f in required:
        if os.path.exists(f):
            print(f"   -> Found: {f}")
        else:
            print(f"   ❌ MISSING: {f} (Please ensure it is in this folder)")
            missing = True
    return not missing

if __name__ == "__main__":
    print("="*40)
    print(" AUTOMATED REPO PREPARATION")
    print("="*40)
    
    create_addon_package()
    print("-" * 20)
    create_readme()
    print("-" * 20)
    if check_docker_files():
        print("="*40)
        print("✅ DONE. You are ready to push to GitHub.")
    else:
        print("="*40)
        print("⚠️  WARNING: Some Docker files are missing.")