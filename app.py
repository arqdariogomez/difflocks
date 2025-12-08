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
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
CURRENT_DIR = Path("/app") # Ruta interna de Docker
sys.path.append(str(CURRENT_DIR))

# Importar lógica interna
try:
    from inference.img2hair import DiffLocksInference
except ImportError:
    sys.path.insert(0, str(CURRENT_DIR))
    from inference.img2hair import DiffLocksInference

# --- RUTAS DE MODELOS (FIX DEFINITIVO) ---
print("🔍 Configurando rutas de modelos...", flush=True)

# 1. Definir la raíz de los checkpoints (según nuestro Dockerfile)
CKPT_ROOT = CURRENT_DIR / "checkpoints"

# 2. Localizar archivos específicos con rutas absolutas
try:
    # El modelo de difusión suele tener nombres variables, buscamos el .pth
    diffusion_dir = CKPT_ROOT / "difflocks_diffusion"
    # Tomamos el primer .pth que encontremos
    PATH_DIFF = list(diffusion_dir.glob("*.pth"))[0]
    
    PATH_CODEC = CKPT_ROOT / "strand_vae" / "strand_codec.pt"
    PATH_RGB = CKPT_ROOT / "rgb2material" / "rgb2material.pt"
    # El config está en el código fuente clonado
    PATH_CONF = CURRENT_DIR / "configs" / "config_scalp_texture_conditional.json"

    print(f"   ✅ Difusión: {PATH_DIFF}", flush=True)
    print(f"   ✅ Codec: {PATH_CODEC}", flush=True)
    print(f"   ✅ Config: {PATH_CONF}", flush=True)

except IndexError:
    print("❌ ERROR CRÍTICO: No hay archivo .pth en /app/checkpoints/difflocks_diffusion", flush=True)
    PATH_DIFF = None
except Exception as e:
    print(f"❌ ERROR DE RUTAS: {e}", flush=True)
    PATH_DIFF = None

OUTPUT_DIR = CURRENT_DIR / "studio_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- INICIALIZACIÓN ---
print("⏳ Cargando Motor IA...", flush=True)
model = None

if PATH_DIFF and PATH_DIFF.exists():
    try:
        torch.set_default_dtype(torch.float32)
        model = DiffLocksInference(
            str(PATH_CODEC),
            str(PATH_CONF),
            str(PATH_DIFF),
            str(PATH_RGB),
            cfg_val=2.5,
            nr_iters_denoise=100,
            nr_chunks_decode=50 
        )
        print("✅ Motor Listo y en GPU.", flush=True)
    except Exception as e:
        print(f"❌ Error iniciando motor: {e}", flush=True)
else:
    print("⚠️ SALTEANDO CARGA DE MOTOR (Faltan archivos)", flush=True)

# --- UTILS ---
def convert_to_obj(npz_path, obj_path):
    if not npz_path.exists(): return
    data = np.load(npz_path)
    pos = data['positions']
    with open(obj_path, 'w') as f:
        f.write(f"# DiffLocks Local Export\no Hair\n")
        v_count = 1
        for strand in pos:
            for p in strand:
                f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
            indices = range(v_count, v_count + len(strand))
            f.write("l " + " ".join(map(str, indices)) + "\n")
            v_count += len(strand)

def generate_plotly_preview(npz_path):
    if not npz_path.exists(): return None
    try:
        data = np.load(npz_path)
        strands = data['positions']
        # Downsampling agresivo para web (mostrar ~1500 pelos)
        step = max(1, int(len(strands) / 1500))
        subset = strands[::step]
        
        # Rotación Y-Up para visualización web
        subset_rot = subset.copy()
        subset_rot[..., 1] = subset[..., 2]
        subset_rot[..., 2] = -subset[..., 1]
        
        n_strands, n_points, _ = subset_rot.shape
        x = np.full((n_strands, n_points + 1), np.nan)
        y = np.full((n_strands, n_points + 1), np.nan)
        z = np.full((n_strands, n_points + 1), np.nan)
        
        x[:, :n_points] = subset_rot[..., 0]
        y[:, :n_points] = subset_rot[..., 1]
        z[:, :n_points] = subset_rot[..., 2]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            mode='lines',
            line=dict(color='#222222', width=2, opacity=0.5),
            hoverinfo='none'
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode='data', bgcolor='rgba(240, 240, 240, 1.0)'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False
        )
        return fig
    except: return None

# --- LOGS ---
class OutputTee(io.StringIO):
    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream
        self.log_content = ""
        self.last_percentage = ""

    def write(self, data):
        self.original_stream.write(data)
        self.log_content += data
        match = re.search(r'(\d+)%', data)
        if match: self.last_percentage = match.group(1) + "%"

    def flush(self): self.original_stream.flush()
    def get_log(self): return self.log_content

tee_stdout = OutputTee(sys.stdout)
tee_stderr = OutputTee(sys.stderr)

def predict(image, cfg_scale):
    if image is None: raise gr.Error("Carga una imagen.")
    if model is None: raise gr.Error("Error Crítico: El modelo no se cargó (revisa logs).")
    
    yield {
        btn_run: gr.Button.update(value="⏳ PROCESANDO...", interactive=False),
        status_box: "🚀 Iniciando...",
        console_box: "Iniciando...",
        result_group: gr.Group.update(visible=False),
        files_out: None,
        plot_out: None
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
        current_status = "Procesando..."
        
        for update in generator:
            current_logs = tee_stdout.get_log() + tee_stderr.get_log()
            pct = f" ({tee_stdout.last_percentage})" if tee_stdout.last_percentage else ""
            
            if isinstance(update, tuple):
                tag = update[0]
                if tag == "status":
                    current_status = update[1]
                    yield {status_box: f"🔄 {current_status}", console_box: current_logs}
                elif tag == "log":
                    if "sampling" in current_status.lower() or "difusión" in current_status.lower():
                         yield {status_box: f"🔄 {current_status}{pct}", console_box: current_logs}
                    else:
                        yield {console_box: current_logs}
                elif tag == "error": raise Exception(update[1])
            
            if len(current_logs) % 50 == 0: yield {console_box: current_logs}

        yield {status_box: "🔨 Generando Geometría..."}
        
        npz_file = current_out / "difflocks_output_strands.npz"
        obj_file = current_out / "hair.obj"
        
        if npz_file.exists() and not obj_file.exists():
            convert_to_obj(npz_file, obj_file)
            
        fig = generate_plotly_preview(npz_file)
        
        outputs = []
        if npz_file.exists(): outputs.append(str(npz_file))
        if obj_file.exists(): outputs.append(str(obj_file))
        
        sys.stdout = tee_stdout.original_stream
        sys.stderr = tee_stderr.original_stream
        
        yield {
            btn_run: gr.Button.update(value="✨ GENERAR PELO", interactive=True),
            status_box: "✅ ¡Generación Exitosa!",
            console_box: tee_stdout.get_log() + tee_stderr.get_log(),
            result_group: gr.Group.update(visible=True),
            files_out: outputs,
            plot_out: fig
        }

    except Exception as e:
        sys.stdout = tee_stdout.original_stream
        sys.stderr = tee_stderr.original_stream
        import traceback
        traceback.print_exc()
        yield {
            btn_run: gr.Button.update(value="❌ Reintentar", interactive=True),
            status_box: f"❌ Error: {str(e)}",
            console_box: tee_stdout.get_log() + f"\n\nERROR:\n{str(e)}"
        }

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="DiffLocks Local") as demo:
    with gr.Row(variant="panel"):
        gr.Markdown("## 💇‍♀️ DiffLocks Studio (Local RTX)")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="filepath", label="Imagen de Entrada", height=320)
            with gr.Group():
                cfg_slider = gr.Slider(1.0, 7.5, value=2.0, step=0.5, label="Fidelity (CFG)")
                btn_run = gr.Button("✨ GENERAR PELO", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Estado", value="Listo.", interactive=False, max_lines=1)
            with gr.Accordion("Logs de Consola", open=False):
                console_box = gr.Code(label="Salida del Sistema", language="shell", interactive=False, lines=8)

            with gr.Group(visible=False) as result_group:
                plot_out = gr.Plot(label="Vista Previa 3D (Plotly)")
                files_out = gr.File(label="Archivos Generados", interactive=False)

    btn_run.click(
        fn=predict,
        inputs=[input_img, cfg_slider],
        outputs=[btn_run, status_box, console_box, result_group, files_out, plot_out]
    )

if __name__ == "__main__":
    print("🚀 Arrancando Servidor Local...", flush=True)
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)