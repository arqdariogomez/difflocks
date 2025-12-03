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

# GENERADOR DE PREDICCIÓN
def predict_stream(image, cfg_scale, export_obj, progress=gr.Progress()):
    if image is None: 
        yield None, None, "⚠️ Por favor sube una imagen primero.", ""
        return

    log_history = ""
    def update_log(msg):
        nonlocal log_history
        log_history += f"{msg}\n"
        return log_history

    try:
        progress(0, desc="Iniciando...")
        gc.collect(); torch.cuda.empty_cache()
        
        # Cargar modelo (puede tardar la primera vez)
        model = load_model()
        
        in_path = PATHS.output_dir / "input_temp.png"
        image.save(in_path)
        
        # Consumimos el generador de img2hair
        # Aquí sucede la magia de la actualización en tiempo real
        iterator = model.file2hair(str(in_path), str(PATHS.output_dir), cfg_val=float(cfg_scale))
        
        strands = None
        
        for msg_type, data1, *data2 in iterator:
            
            if msg_type == "status":
                # Actualizar barra de progreso (Texto)
                progress(0.5, desc=data1) # El 0.5 es dummy, lo importante es el texto
                yield None, None, data1, update_log(f"[{msg_type.upper()}] {data1}")
                
            elif msg_type == "log":
                # Agregar a consola
                yield None, None, "Procesando...", update_log(f"   > {data1}")
                
            elif msg_type == "error":
                yield None, None, f"❌ Error: {data1}", update_log(f"❌ ERROR: {data1}")
                return
                
            elif msg_type == "result":
                strands = data1
                
        # Proceso finalizado
        if strands is None: 
            yield None, None, "❌ Fallo: No se generaron hebras.", log_history
            return
        
        out_npz = str(PATHS.output_dir / "difflocks_output_strands.npz")
        out_obj = None
        
        if export_obj:
            progress(0.9, desc="Exportando OBJ...")
            yield None, None, "Generando archivo OBJ...", update_log("💾 Convirtiendo a OBJ...")
            
            out_obj_path = PATHS.output_dir / "hair.obj"
            pos = strands.cpu().numpy()
            with open(out_obj_path, 'w') as f:
                f.write("o Hair\n")
                v_off = 1
                for s in pos:
                    for p in s: f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
                    indices = range(v_off, v_off + len(s))
                    f.write("l " + " ".join(map(str, indices)) + "\n")
                    v_off += len(s)
            out_obj = str(out_obj_path)
            
        update_log("✨ ¡Listo!")
        yield out_npz, out_obj, "✅ ¡Generación Exitosa!", log_history
        
    except Exception as e:
        traceback.print_exc()
        yield None, None, f"❌ Error Crítico: {str(e)}", log_history + f"\n❌ EXCEPTION: {str(e)}"

# DISEÑO DE LA INTERFAZ
with gr.Blocks(theme=gr.themes.Base(), title="DiffLocks Studio") as app:
    with gr.Row():
        gr.Markdown("""
        # 💇‍♀️ DiffLocks Studio (T4 Compatible)
        **Generación de Pelo 3D de Alta Fidelidad** | *Fork Optimizado para Kaggle*
        """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Image", height=350)
            
            with gr.Group():
                cfg = gr.Slider(1.5, 7.5, value=2.5, step=0.1, label="Fidelidad (CFG Scale)", 
                              info="2.5 es recomendado. Mayor valor = más contraste.")
                chk_obj = gr.Checkbox(value=True, label="Generar archivo .OBJ (Para Blender)")
            
            btn = gr.Button("✨ Generar Pelo 3D", variant="primary", scale=1)
            
        with gr.Column(scale=1):
            status = gr.Label(value="Listo para empezar", label="Estado Actual")
            
            with gr.Row():
                file_npz = gr.File(label="Datos Crudos (.npz)")
                file_obj = gr.File(label="Modelo 3D (.obj)")
            
            # Consola desplegable
            with gr.Accordion("📜 Consola de Procesos", open=False):
                console_log = gr.Textbox(label="Logs Detallados", lines=10, interactive=False)

    # Evento
    btn.click(
        predict_stream, 
        inputs=[input_img, cfg, chk_obj], 
        outputs=[file_npz, file_obj, status, console_log]
    )

if __name__ == "__main__":
    # Inline=True para que salga en el notebook
    app.queue().launch(share=True, inline=True, allowed_paths=[str(PATHS.output_dir)], debug=True)
