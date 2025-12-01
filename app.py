import sys
import os
import gradio as gr
from pathlib import Path
import torch
import traceback

# Setup Paths
sys.path.insert(0, os.getcwd())

print("⏳ Initializing DiffLocks App...")

try:
    from config import PATHS, PLATFORM
    from inference.img2hair import DiffLocksInference
except Exception as e:
    print("❌ FATAL ERROR DURING IMPORT:")
    traceback.print_exc()
    raise e

# --- FIX: Asegurar output dir antes de lanzar ---
try:
    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {PATHS.output_dir}")
except Exception as e:
    print(f"⚠️ Warning: Could not create output dir: {e}")

MODEL_INSTANCE = None

def load_model():
    global MODEL_INSTANCE
    if MODEL_INSTANCE is not None:
        return MODEL_INSTANCE
    
    print("⏳ Loading Model Checkpoints...")
    ckpt_root = PATHS.models_dir
    strand_codec = ckpt_root / "strand_vae/strand_codec.pt"
    diffusion = list(ckpt_root.glob("difflocks_diffusion/*.pth"))
    rgb2mat = ckpt_root / "rgb2material/rgb2material.pt"
    config_path = Path("configs/config_scalp_texture_conditional.json")
    
    if not diffusion:
        raise FileNotFoundError(f"Diffusion checkpoint not found in {ckpt_root}")
        
    MODEL_INSTANCE = DiffLocksInference(
        str(strand_codec),
        str(config_path),
        str(diffusion[0]),
        str(rgb2mat) if rgb2mat.exists() else None,
        cfg_val=4.0
    )
    print("✅ Model Loaded Successfully!")
    return MODEL_INSTANCE

def predict(image, cfg_scale, export_obj):
    if image is None: return None, None, "Please upload an image"
    try:
        PATHS.output_dir.mkdir(parents=True, exist_ok=True)
        model = load_model()
        model.cfg_val = float(cfg_scale)
        
        in_path = PATHS.output_dir / "input_temp.png"
        image.save(in_path)
        
        strands, mats = model.file2hair(str(in_path), str(PATHS.output_dir))
        
        if strands is None: return None, None, "Inference returned None (Face not detected?)"
            
        out_npz = PATHS.output_dir / "difflocks_output_strands.npz"
        out_obj = None
        
        if export_obj:
            out_obj = PATHS.output_dir / "hair.obj"
            pos = strands.cpu().numpy()
            with open(out_obj, 'w') as f:
                f.write("o Hair\n")
                v_off = 1
                for s in pos:
                    for p in s: f.write(f"v {p[0]} {p[1]} {p[2]}\n")
                    indices = range(v_off, v_off + len(s))
                    f.write("l " + " ".join(map(str, indices)) + "\n")
                    v_off += len(s)
                    
        return str(out_npz), str(out_obj) if out_obj else None, "Success!"
        
    except Exception as e:
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"

# UI configuration
with gr.Blocks(title="DiffLocks Studio", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 💇‍♀️ DiffLocks Studio")
    gr.Markdown(f"Running on: **{PLATFORM.name}**")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            cfg = gr.Slider(1.5, 7.5, value=4.0, label="CFG Scale")
            chk_obj = gr.Checkbox(value=True, label="Export OBJ")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="Status")
            file_npz = gr.File(label="NPZ Data")
            file_obj = gr.File(label="OBJ Model")
            
    btn.click(predict, [input_img, cfg, chk_obj], [file_npz, file_obj, status])

if __name__ == "__main__":
    # Settings para estabilidad en Kaggle
    app.launch(share=True, allowed_paths=[str(PATHS.output_dir)], debug=True, inline=False)
