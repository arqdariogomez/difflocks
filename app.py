
import os
import sys
import time
import shutil
import subprocess
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. SETUP ---
try:
    from platform_config import cfg
except ImportError:
    sys.path.append(".")
    from platform_config import cfg

if str(cfg.repo_dir) not in sys.path:
    sys.path.append(str(cfg.repo_dir))

if cfg.platform == 'huggingface' and not cfg.blender_exe.exists():
    print("📦 Downloading Blender for HF Space...")
    b_dir = Path("/tmp/blender")
    b_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run("wget -q -O /tmp/blender.tar.xz https://download.blender.org/release/Blender4.2/blender-4.2.5-linux-x64.tar.xz", shell=True)
    subprocess.run(f"tar -xf /tmp/blender.tar.xz -C {b_dir} --strip-components=1", shell=True)

# --- 2. MODEL LOADING ---
try:
    import spaces
    has_zerogpu = True
except ImportError:
    has_zerogpu = False

print(f"🚀 Initializing on {cfg.platform} (GPU: {cfg.has_gpu}, ZeroGPU: {has_zerogpu})")

from inference.img2hair import DiffLocksInference

# Locate Weights
ckpt_dir = cfg.repo_dir / "checkpoints"
diff_path = list((ckpt_dir/"difflocks_diffusion").glob("*.pth"))
vae_path = ckpt_dir / "strand_vae/strand_codec.pt"
conf_path = cfg.repo_dir / "configs/config_scalp_texture_conditional.json"

model = None

def load_model():
    global model
    if model is not None: return
    
    if not diff_path or not vae_path.exists():
        # Fallback recursive search
        print("⚠️ Standard paths failed. Searching recursively...")
        found_diff = list(cfg.repo_dir.rglob("*diffusion*.pth"))
        found_vae = list(cfg.repo_dir.rglob("strand_codec.pt"))
        
        if not found_diff or not found_vae:
            raise FileNotFoundError("Checkpoints missing! Please download them.")
        
        diff_p = str(found_diff[0])
        vae_p = str(found_vae[0])
    else:
        diff_p = str(diff_path[0])
        vae_p = str(vae_path)

    print(f"Loading Model: {Path(diff_p).name}")
    model = DiffLocksInference(vae_p, str(conf_path), diff_p, None)

if not has_zerogpu and cfg.has_gpu:
    try: load_model() 
    except: print("⚠️ Could not preload model (maybe missing files).")

# --- 3. LOGIC ---
def run_inference(img, cfg_val, fmts):
    if img is None: raise gr.Error("Image required")
    if model is None: load_model()
    
    job_id = f"j_{int(time.time())}"
    job_dir = cfg.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    img_p = job_dir / "input.png"
    if isinstance(img, str): shutil.copy(img, img_p)
    else: img.save(img_p)
    
    # 1. GENERATION
    model.cfg_val = float(cfg_val)
    
    for update in model.file2hair(str(img_p), str(job_dir)):
        if isinstance(update, tuple) and update[0] == "status":
            yield None, None, f"⚙️ {update[1]}", gr.Group(visible=False)
            
    npz_path = job_dir / "difflocks_output_strands.npz"
    yield None, None, "🎨 Rendering Preview...", gr.Group(visible=False)
    
    # 2. PREVIEW
    prev_path = None
    fig_3d = None
    try:
        d=np.load(npz_path)['positions']
        s=d[::max(1,len(d)//20000)]; p=s.reshape(-1,3); x,y,z=p[:,0],p[:,1],p[:,2]; rx,ry,rz=x,-z,y
        plt.style.use('dark_background'); f,ax=plt.subplots(1,3,figsize=(18,6)); f.patch.set_facecolor('#111')
        for a,u,v,d_ in zip(ax,[rx,ry,rx],[rz,rz,ry],[ry,np.abs(rx),rz]):
            idx=np.argsort(d_); a.scatter(u[idx],v[idx],c=(d_[idx]-d_.min())/(d_.max()-d_.min()+1e-8),cmap='copper',s=0.5,lw=0); a.axis('off')
        prev_path = job_dir/"prev.png"
        f.savefig(prev_path, facecolor='#111', bbox_inches='tight'); plt.close(f)
        
        s3=d[::max(1,len(d)//30000)][:,::8,:]; p3=s3.reshape(-1,3); x3,y3,z3=p3[:,0],p3[:,1],p3[:,2]; rx3,ry3,rz3=x3,-z3,y3
        c3=np.hstack([np.tile(np.linspace(0.3,1,s3.shape[1]),(s3.shape[0],1)),np.zeros((s3.shape[0],1))]).flatten()
        fig_3d = go.Figure(data=[go.Scatter3d(x=np.hstack([rx3.reshape(s3.shape[:2]),np.full((s3.shape[0],1),np.nan)]).flatten(), y=np.hstack([ry3.reshape(s3.shape[:2]),np.full((s3.shape[0],1),np.nan)]).flatten(), z=np.hstack([rz3.reshape(s3.shape[:2]),np.full((s3.shape[0],1),np.nan)]).flatten(), mode='lines', line=dict(width=1.5,color=c3,colorscale=[[0,'#505050'],[1,'white']],showscale=False))], layout=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),bgcolor='rgba(0,0,0,0)'),margin=dict(l=0,r=0,b=0,t=0),height=500))
    except Exception as e: print(f"Viz Error: {e}")

    # 3. EXPORT
    outputs = [str(npz_path)]
    blender_keys = [k for k,v in {'.blend':'Blender','.abc':'Alembic','.usd':'USD'}.items() if v in fmts]
    
    if blender_keys and cfg.blender_exe.exists():
        yield None, None, "🟧 Running Blender Export...", gr.Group(visible=False)
        script = cfg.repo_dir / "inference/converter_blender.py"
        cmd = [str(cfg.blender_exe), "-b", "-P", str(script), "--", str(npz_path), str(job_dir/"hair")] + [k.replace('.','') for k in blender_keys]
        subprocess.run(cmd)
        for k in blender_keys:
            f = job_dir / f"hair{k}"
            if f.exists(): outputs.append(str(f))
            
    import zipfile
    zip_f = job_dir / "results.zip"
    with zipfile.ZipFile(zip_f, 'w') as z:
        for f in outputs: z.write(f, Path(f).name)
    
    yield fig_3d, str(prev_path), "✅ Done!", gr.Group(visible=True), str(zip_f)

if has_zerogpu:
    run_inference = spaces.GPU(duration=120)(run_inference)

# --- 4. UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="DiffLocks Studio") as demo:
    gr.Markdown("# 💇‍♀️ DiffLocks Studio (Universal)")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="filepath", label="Input Face")
            cfg_s = gr.Slider(1, 7, 2.5, label="CFG Scale")
            chk = gr.CheckboxGroup(["Blender", "Alembic", "USD"], label="Exports (Requires Blender)", value=[])
            btn = gr.Button("🚀 Generate", variant="primary")
        with gr.Column():
            status = gr.HTML("Ready")
            with gr.Group(visible=False) as res_grp:
                with gr.Tabs():
                    with gr.Tab("3D"): p3d = gr.Plot()
                    with gr.Tab("2D"): p2d = gr.Image()
                files = gr.File(label="Download ZIP")
    btn.click(run_inference, [inp, cfg_s, chk], [p3d, p2d, status, res_grp, files])

if __name__ == "__main__":
    demo.queue().launch(share=cfg.needs_share, server_name="0.0.0.0", server_port=7860)
