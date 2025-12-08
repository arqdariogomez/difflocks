# DiffLocks Studio (Docker Edition)

**High-Fidelity 3D Hair Generation from Single Images.**
*Fork optimized for Local usage (Docker) and Google Colab.*

## 🚀 Features
*   **One-Click Docker Setup:** Run locally on your GPU without messing up your Python environment.
*   **Native 3D Preview:** Visualize results directly in the browser via Plotly.
*   **Blender Integration:** Includes a custom Add-on to import the generated `.npz` hair strands.
*   **Reactive UI:** Real-time console logs and progress tracking.

---

## 🛠️ Installation (Local Docker)

### Prerequisites
*   **Docker Desktop** installed and running.
*   **NVIDIA GPU** (RTX 2060 or higher recommended).
*   **NVIDIA Drivers** updated.

### 1. Setup
Clone this repository and navigate to the folder:
```bash
git clone https://github.com/arqdariogomez/difflocks.git
cd difflocks
```

### 2. Prepare Checkpoints
You need to download the model weights from the [Meshcapade website](https://difflocks.is.tue.mpg.de).
Place them in the `checkpoints/` folder following this exact structure:

```text
checkpoints/
├── difflocks_diffusion/
│   └── scalp_v9_40k_06730000.pth
├── rgb2material/
│   └── rgb2material.pt
└── strand_vae/
    └── strand_codec.pt
```

### 3. Run
Open a terminal in the folder and run:

```bash
docker compose up --build
```

Wait for the installation to finish. Once ready, open your browser at:
**http://localhost:7860**

---

## 🔌 Blender Add-on

To import the generated hair into Blender:

1.  Go to the `blender_addon/` folder in this repo.
2.  Download `DiffLocks_Blender_Importer.zip`.
3.  In Blender: `Edit > Preferences > Add-ons > Install...` and select the zip.
4.  Enable the add-on.
5.  Import via: `File > Import > DiffLocks Hair (.npz)`.

---

## ☁️ Google Colab
You can also run this project in the cloud using the provided Notebook.