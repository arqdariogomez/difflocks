import os
import torch
import numpy as np
import k_diffusion as K
import torchvision.transforms as T
import torchvision
import traceback
import gc
import time
import psutil
import cv2
import mediapipe as mp

# Imports internos del repositorio
from models.strand_codec import StrandCodec
from models.rgb_to_material import RGB2MaterialModel
from utils.diffusion_utils import sample_images_cfg
from utils.strand_util import sample_strands_from_scalp_with_density
from data_loader.dataloader import DiffLocksDataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constantes globales
DEFAULT_BODY_DATA_DIR = "data_loader/difflocks_bodydata"
tbn_space_to_world_cpu = torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]).float()

# --- UTILIDADES DE MEMORIA ---
def get_memory_info():
    """Monitor de RAM y VRAM."""
    try:
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)
        ram_total = ram.total / (1024**3)
        gpu_used = 0
        gpu_total = 0
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return ram_used, ram_total, gpu_used, gpu_total
    except:
        return 0,0,0,0

def log_memory(phase):
    r_u, r_t, g_u, g_t = get_memory_info()
    print(f"   [MEM] {phase}: RAM {r_u:.1f}/{r_t:.1f}GB | GPU {g_u:.1f}/{g_t:.1f}GB")

def force_cleanup():
    """Limpieza profunda de Garbage Collector y VRAM."""
    gc.collect()
    gc.collect() # Doble pasada para referencias circulares
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    time.sleep(0.5) # Pausa para permitir al OS recuperar recursos

# --- UTILIDADES DE IMAGEN (CROP_FACE RESTAURADO) ---
def crop_face(image, face_landmarks, output_size=770, crop_size_multiplier=2.8):
    h, w, _ = image.shape
    xs = [l.x for l in face_landmarks]; ys = [l.y for l in face_landmarks]
    min_x, max_x = min(xs) * w, max(xs) * w
    min_y, max_y = min(ys) * h, max(ys) * h
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    face_w, face_h = max_x - min_x, max_y - min_y
    size = max(face_w, face_h) * 1.5
    x1 = int(cx - size / 2); y1 = int(cy - size / 2)
    x2 = int(cx + size / 2); y2 = int(cy + size / 2)
    pad_l = max(0, -x1); pad_t = max(0, -y1)
    pad_r = max(0, x2 - w); pad_b = max(0, y2 - h)
    if any([pad_l, pad_t, pad_r, pad_b]):
        image = np.pad(image, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant')
        x1 += pad_l; y1 += pad_t; x2 += pad_l; y2 += pad_t
    crop = image[y1:y2, x1:x2]
    try: return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    except: return cv2.resize(image, (output_size, output_size))

# --- CLASE MEDIAPIPE (RESTAURADA) ---
class Mediapipe:
    def __init__(self):
        # Aseguramos que el path al asset sea correcto relativo al repo
        asset_path = 'inference/assets/face_landmarker_v2_with_blendshapes.task'
        if not os.path.exists(asset_path):
             # Fallback por si la ruta de ejecución varía
             asset_path = os.path.join(os.getcwd(), asset_path)
        
        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, 
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True, 
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def run(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        res = self.detector.detect(mp_image)
        return (res.face_blendshapes[0], res.face_landmarks[0]) if res.face_landmarks else (None, None)

# --- MOTOR DE INFERENCIA PRINCIPAL ---
class DiffLocksInference():
    def __init__(self, path_ckpt_strandcodec, path_config_difflocks, path_ckpt_difflocks,
                 path_ckpt_rgb2material=None, cfg_val=1.0, nr_iters_denoise=100, nr_chunks_decode=150):
        
        # PARAMETROS CRITICOS
        self.nr_chunks_decode_strands = nr_chunks_decode # Aumentado a 150 para reducir pico de RAM
        self.nr_iters_denoise = nr_iters_denoise
        self.cfg_val = cfg_val
        
        # RUTAS
        self.path_ckpt_strandcodec = path_ckpt_strandcodec
        self.path_config_difflocks = path_config_difflocks
        self.path_ckpt_difflocks = path_ckpt_difflocks
        self.path_ckpt_rgb2material = path_ckpt_rgb2material
        
        # INICIALIZACION
        self.mediapipe_img = Mediapipe()
        self.normalization_dict = DiffLocksDataset.get_normalization_data()
        
        scalp_path = os.path.join(DEFAULT_BODY_DATA_DIR, "scalp.ply")
        if not os.path.exists(scalp_path): 
            scalp_path = "data_loader/difflocks_bodydata/scalp.ply"
            
        self.scalp_trimesh, self.scalp_mesh_data = DiffLocksDataset.compute_scalp_data(scalp_path)
        self.tbn_space_to_world = tbn_space_to_world_cpu
        
        # CACHE CPU (Para evitar mover datos grandes repetidamente)
        self.norm_dict_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.normalization_dict.items()}
        self.mesh_data_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.scalp_mesh_data.items()}

    def rgb2hair(self, rgb_img, out_path=None):
        if out_path: os.makedirs(out_path, exist_ok=True)
        log_memory("Inicio Inferencia")
        
        try:
            # ---------------------------------------------------------
            # 1. DETECCION FACIAL
            # ---------------------------------------------------------
            print("   [1/4] Procesando Geometría Facial...")
            # Convertir Tensor GPU -> Numpy CPU para Mediapipe
            frame = (rgb_img.permute(0,2,3,1).squeeze(0)*255).byte().cpu().numpy()
            _, lms = self.mediapipe_img.run(frame)
            
            if not lms: 
                print("   [ERROR] No se detectó rostro.")
                return None, None
            
            cropped_face = crop_face(frame, lms, 770)
            del frame # Liberar imagen original grande
            
            # Re-subir crop a GPU normalizado
            rgb_img_gpu = torch.tensor(cropped_face).cuda().permute(2,0,1).unsqueeze(0).float()/255.0
            rgb_img_cpu = rgb_img_gpu.cpu().clone() # Backup para guardar al final
            
            log_memory("Post-Face")

            # ---------------------------------------------------------
            # 2. DINO FEATURES (GPU)
            # ---------------------------------------------------------
            print("   [2/4] Extrayendo características DINO...")
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', verbose=False).cuda()
            tf = T.Compose([T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
            
            with torch.no_grad(): 
                out = dinov2.forward_features(tf(rgb_img_gpu))
            
            patch = out["x_norm_patchtokens"]
            cls_tok = out["x_norm_clstoken"]
            h = w = int(patch.shape[1]**0.5)
            patch_emb = patch.reshape(patch.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            
            # Guardar resultados en CPU
            patch_emb_cpu = patch_emb.cpu().clone()
            cls_tok_cpu = cls_tok.cpu().clone()
            
            # LIMPIEZA TOTAL DINO (Liberar ~1.5GB VRAM)
            del dinov2, out, patch, cls_tok, patch_emb, rgb_img_gpu
            force_cleanup()
            log_memory("Post-DINO Cleanup")

            # ---------------------------------------------------------
            # 3. DIFUSION (GPU - Fase Pesada)
            # ---------------------------------------------------------
            print("   [3/4] Generando Textura Neural (Diffusion FP32)...")
            
            conf = K.config.load_config(self.path_config_difflocks)
            model = K.config.make_denoiser_wrapper(conf)(K.config.make_model(conf).cuda())
            
            # Cargar pesos con seguridad (weights_only=False silencia warnings)
            ckpt = torch.load(self.path_ckpt_difflocks, map_location='cpu', weights_only=False)
            model.inner_model.load_state_dict(ckpt['model_ema'])
            del ckpt
            force_cleanup()
            
            model.eval()
            model.inner_model.condition_dropout_rate = 0.0
            
            # Subir latents a GPU solo para este paso
            extra = {
                'latents_dict': {
                    "dinov2": {
                        "cls_token": cls_tok_cpu.cuda(), 
                        "final_latent": patch_emb_cpu.cuda()
                    }
                }
            }
            
            scalp = sample_images_cfg(1, self.cfg_val, [-1., 10000.], model, conf['model'], self.nr_iters_denoise, extra)
            
            # BAJAR RESULTADO A CPU INMEDIATAMENTE
            scalp_cpu = scalp.cpu().clone()
            sigma_data = conf['model']["sigma_data"]
            
            # LIMPIEZA TOTAL DIFUSION (Liberar ~4GB VRAM)
            del model, scalp, extra, conf
            force_cleanup()
            
            print("   [INFO] GPU liberada completamente para Decoding")
            log_memory("Post-Diffusion Cleanup")

            # Calcular densidad
            density = (scalp_cpu[:,-1:]*(0.5/sigma_data)+0.5).clamp(0,1)
            density[density<0.02] = 0.0
            
            if density.sum() == 0:
                print("   [ERROR] Mapa de densidad vacío.")
                return None, None

            # ---------------------------------------------------------
            # 4. DECODING (CPU - Fase Crítica de RAM)
            # ---------------------------------------------------------
            print("   [4/4] Decodificando Strands (Modo CPU, Chunked)...")
            log_memory("Pre-Codec Load")
            
            # Cargar Codec en CPU
            codec = StrandCodec(do_vae=False, decode_type="dir", nr_verts_per_strand=256).cpu()
            codec.load_state_dict(torch.load(self.path_ckpt_strandcodec, map_location='cpu', weights_only=False))
            codec.eval()
            
            log_memory("Post-Codec Load")
            
            print(f"   [INFO] Usando {self.nr_chunks_decode_strands} chunks para proteger RAM...")
            
            # INFERENCIA DE STRANDS EN CPU
            strands, _ = sample_strands_from_scalp_with_density(
                scalp_cpu[:,0:-1], 
                density, 
                codec, 
                self.norm_dict_cpu, 
                self.mesh_data_cpu, 
                self.tbn_space_to_world.cpu(), 
                self.nr_chunks_decode_strands
            )
            
            del codec, scalp_cpu, density
            force_cleanup()
            log_memory("Post-Decoding")

            # ---------------------------------------------------------
            # 5. GUARDADO
            # ---------------------------------------------------------
            print("   [5/5] Guardando resultados...")
            
            if out_path and strands is not None:
                # Usar compresión para ahorrar espacio en disco
                positions = strands.cpu().numpy()
                np.savez_compressed(
                    os.path.join(out_path, "difflocks_output_strands.npz"), 
                    positions=positions
                )
                del positions
                
                # Guardar imagen RGB de referencia
                torchvision.utils.save_image(rgb_img_cpu, os.path.join(out_path, "rgb.png"))
            
            print("   ✅ [EXITO] Inferencia completada.")
            log_memory("Final")
            
            # Devolvemos None en el segundo argumento (materiales) para ahorrar RAM
            return strands, None

        except Exception as e:
            print(f"   ❌ [ERROR CRITICO] {e}")
            traceback.print_exc()
            log_memory("Error State")
            raise
        finally:
            force_cleanup()

    def file2hair(self, fpath, out):
        img = cv2.imread(fpath)
        if img is None: raise FileNotFoundError(f"{fpath}")
        rgb = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda().permute(2,0,1).unsqueeze(0).float()/255.
        return self.rgb2hair(rgb, out)
