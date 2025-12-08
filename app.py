import gradio as gr
import torch
import os
import sys
import re
import numpy as np
from pathlib import Path
import shutil
import time
import io
import subprocess

# --- CONFIG ---
CURRENT_DIR = Path(os.getcwd())
sys.path.append(str(CURRENT_DIR))

try:
    from inference.img2hair import DiffLocksInference
except ImportError:
    sys.path.insert(0, str(CURRENT_DIR))
    from inference.img2hair import DiffLocksInference

# --- CHECKPOINTS ---
print("🔍 Scanning models...", flush=True)
def find_file(name, search_path):
    found = list(search_path.rglob(name))
    if found: return str(found[0])
    return None

PATH_DIFF = find_file("*scalp*.pth", CURRENT_DIR)
PATH_CODEC = find_file("strand_codec.pt", CURRENT_DIR)
PATH_RGB = find_file("rgb2material.pt", CURRENT_DIR)
PATH_CONF = find_file("config_scalp_texture_conditional.json", CURRENT_DIR)

if not all([PATH_DIFF, PATH_CODEC, PATH_CONF]):
    print("❌ ERROR: Checkpoints missing.", flush=True)
else:
    print("✅ Checkpoints OK.", flush=True)

OUTPUT_DIR = CURRENT_DIR / "studio_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- ENGINE ---
print("⏳ Initializing...", flush=True)
try:
    torch.set_default_dtype(torch.float32)
    model = DiffLocksInference(
        PATH_CODEC, PATH_CONF, PATH_DIFF, PATH_RGB,
        cfg_val=2.5, nr_iters_denoise=100, nr_chunks_decode=50 
    )
    print("✅ Engine Ready.", flush=True)
except Exception as e:
    print(f"❌ Init Failed: {e}", flush=True)
    model = None

# --- BLENDER BRIDGE ---
def run_blender_conversion(npz_path, output_base):
    script_path = "/app/converter.py"
    if not os.path.exists(script_path): return False, "Script missing"

    cmd = ["blender", "-b", "-P", script_path, "--", str(npz_path), str(output_base)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0: return True, "Success"
        else: return False, f"Blender Error: {result.stderr}"
    except Exception as e: return False, str(e)

# --- LOGS ---
class OutputTee(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.log_content = ""
        self.last_pct = ""
    def write(self, data):
        self.stream.write(data)
        self.log_content += data
        match = re.search(r'(\d+)%', data)
        if match: self.last_pct = match.group(1) + "%"
    def flush(self): self.stream.flush()
    def get_log(self): return self.log_content

tee_stdout = OutputTee(sys.stdout)
tee_stderr = OutputTee(sys.stderr)

def predict(image, cfg_scale, do_blender):
    if image is None: raise gr.Error("No image.")
    
    yield {
        btn_run: gr.Button.update(value="⏳ PROCESSING...", interactive=False),
        status_box: "🚀 Starting...",
        console_box: "Init...",
        result_group: gr.Group.update(visible=False),
        files_out: None,
        viewer: None
    }
    
    job_id = str(int(time.time()))
    current_out = OUTPUT_DIR / job_id
    current_out.mkdir(parents=True, exist_ok=True)
    model.cfg_val = float(cfg_scale)
    
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    tee_stdout.log_content = ""
    tee_stderr.log_content = ""
    
    try:
        generator = model.file2hair(str(image), str(current_out))
        current_status = "Processing..."
        
        for update in generator:
            current_logs = tee_stdout.get_log() + tee_stderr.get_log()
            pct = f" ({tee_stdout.last_pct})" if tee_stdout.last_pct else ""
            
            if isinstance(update, tuple):
                tag = update[0]
                if tag == "status":
                    current_status = update[1]
                    yield {status_box: f"🔄 {current_status}", console_box: current_logs}
                elif tag == "log":
                    if "sampling" in current_status.lower():
                         yield {status_box: f"🔄 {current_status}{pct}", console_box: current_logs}
                    else:
                        yield {console_box: current_logs}
                elif tag == "error": raise Exception(update[1])
            
            if len(current_logs) % 50 == 0: yield {console_box: current_logs}

        # Conversion
        npz_file = current_out / "difflocks_output_strands.npz"
        outputs = []
        viewer_file = None
        
        if npz_file.exists():
            outputs.append(str(npz_file))
            if do_blender:
                yield {status_box: "🔨 Running Blender Conversion..."}
                base_name = current_out / "hair"
                success, msg = run_blender_conversion(npz_file, base_name)
                if success:
                    for ext in [".blend", ".abc", ".glb"]:
                        f = current_out / f"hair{ext}"
                        if f.exists():
                            outputs.append(str(f))
                            if ext == ".glb": viewer_file = str(f)
                else:
                    print(f"Blender failed: {msg}")

        sys.stdout = tee_stdout.stream
        sys.stderr = tee_stderr.stream
        
        yield {
            btn_run: gr.Button.update(value="✨ GENERATE HAIR", interactive=True),
            status_box: "✅ Done!",
            console_box: tee_stdout.get_log() + tee_stderr.get_log(),
            result_group: gr.Group.update(visible=True),
            files_out: outputs,
            viewer: viewer_file
        }

    except Exception as e:
        sys.stdout = tee_stdout.stream
        sys.stderr = tee_stderr.stream
        yield {
            btn_run: gr.Button.update(value="❌ Retry", interactive=True),
            status_box: f"Error: {e}",
            console_box: tee_stdout.get_log() + f"\nERROR:\n{e}"
        }

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="DiffLocks Local") as demo:
    with gr.Row(variant="panel"):
        gr.Markdown("## 💇‍♀️ DiffLocks Studio (Local RTX)")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="Input Reference", height=320)
            with gr.Group():
                cfg_slider = gr.Slider(1.0, 7.5, value=2.0, step=0.5, label="Fidelity Strength")
                check_blender = gr.Checkbox(label="📦 Export Formats (.blend, .abc, .glb)", value=True)
                btn_run = gr.Button("✨ GENERATE HAIR", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Status", interactive=False, max_lines=1)
            with gr.Accordion("Console Logs", open=False):
                console_box = gr.Code(label="System Output", language="shell", interactive=False, lines=8)

            with gr.Group(visible=False) as result_group:
                viewer = gr.Model3D(clear_color=[0.8, 0.8, 0.8, 1.0], label="3D Preview (.glb)", interactive=True, height=500)
                files_out = gr.File(label="Generated Files", interactive=False)

    btn_run.click(fn=predict, inputs=[input_img, cfg_slider, check_blender], outputs=[btn_run, status_box, console_box, result_group, files_out, viewer])

if __name__ == "__main__":
    print("🚀 Starting...", flush=True)
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)