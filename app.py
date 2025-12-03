import sys, os, gradio as gr, torch, traceback, gc
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"
sys.path.insert(0, os.getcwd())

try:
    from config import PATHS, PLATFORM
    from inference.img2hair import DiffLocksInference
    PATHS.output_dir.mkdir(parents=True, exist_ok=True)
except Exception as e:
    traceback.print_exc(); raise e

MODEL_INSTANCE = None

def load_model():
    global MODEL_INSTANCE
    if MODEL_INSTANCE: return MODEL_INSTANCE
    print("⏳ Loading Model...")
    ckpt_root = PATHS.models_dir
    strand_codec = ckpt_root / "strand_vae/strand_codec.pt"
    diffusion = list(ckpt_root.glob("difflocks_diffusion/*.pth"))
    rgb2mat = ckpt_root / "rgb2material/rgb2material.pt"
    config_path = Path("configs/config_scalp_texture_conditional.json")
    gc.collect(); torch.cuda.empty_cache()
    MODEL_INSTANCE = DiffLocksInference(
        str(strand_codec), str(config_path), str(diffusion[0]),
        str(rgb2mat) if rgb2mat.exists() else None, cfg_val=4.0, nr_chunks_decode=150
    )
    return MODEL_INSTANCE

def predict(image, cfg_scale, export_obj):
    if image is None: return None, None, "⚠️ Please upload an image"
    try:
        gc.collect(); torch.cuda.empty_cache()
        model = load_model()
        
        in_path = PATHS.output_dir / "input_temp.png"
        image.save(in_path)
        
        print(f"🔄 UI Request: CFG={cfg_scale}")
        strands, _ = model.file2hair(str(in_path), str(PATHS.output_dir), cfg_val=float(cfg_scale))
        
        if strands is None: return None, None, "❌ Face not detected"
        
        out_npz = str(PATHS.output_dir / "difflocks_output_strands.npz")
        out_obj = None
        if export_obj:
            out_obj_path = PATHS.output_dir / "hair.obj"
            pos = strands.cpu().numpy()
            with open(out_obj_path, 'w') as f:
                f.write("o Hair\n")
                v_off = 1
                for s in pos:
                    for p in s: f.write(f"v {p[0]} {p[1]} {p[2]}\n")
                    indices = range(v_off, v_off + len(s))
                    f.write("l " + " ".join(map(str, indices)) + "\n")
                    v_off += len(s)
            out_obj = str(out_obj_path)
            
        return out_npz, out_obj, "✅ Success! Download files below."
        
    except Exception as e:
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"

with gr.Blocks(title="DiffLocks Studio", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 💇‍♀️ DiffLocks Studio")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            cfg = gr.Slider(1.5, 7.5, value=4.0, step=0.1, label="CFG Scale (Fidelity)")
            chk_obj = gr.Checkbox(value=True, label="Export OBJ")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="Status")
            file_npz = gr.File(label="NPZ Data")
            file_obj = gr.File(label="OBJ Model")
            
    btn.click(predict, [input_img, cfg, chk_obj], [file_npz, file_obj, status])

if __name__ == "__main__":
    # --- FIX TIMEOUT: ACTIVAMOS QUEUE ---
    app.queue(max_size=5) 
    app.launch(share=True, allowed_paths=[str(PATHS.output_dir)], debug=True, inline=False)
