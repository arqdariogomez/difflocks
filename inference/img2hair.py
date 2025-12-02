import os
import torch
import numpy as np
import k_diffusion as K
import torchvision.transforms as T
import torchvision
from models.strand_codec import StrandCodec
from models.rgb_to_material import RGB2MaterialModel
from utils.diffusion_utils import sample_images_cfg
from utils.strand_util import sample_strands_from_scalp_with_density
from data_loader.dataloader import DiffLocksDataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
import gc
import time

DEFAULT_BODY_DATA_DIR = "data_loader/difflocks_bodydata"
tbn_space_to_world_cpu = torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]).float()

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

class Mediapipe:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='inference/assets/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                             output_facial_transformation_matrixes=True, num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
    def run(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        res = self.detector.detect(mp_image)
        return (res.face_blendshapes[0], res.face_landmarks[0]) if res.face_landmarks else (None, None)

def force_cleanup():
    """Liberación agresiva de memoria GPU y RAM."""
    gc.collect()
    gc.collect()  # Doble llamada para objetos con referencias circulares
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Resetear estadísticas para liberar memoria fragmentada
        torch.cuda.reset_peak_memory_stats()
    time.sleep(0.5)  # Dar tiempo al sistema

class DiffLocksInference():
    def __init__(self, path_ckpt_strandcodec, path_config_difflocks, path_ckpt_difflocks,
                 path_ckpt_rgb2material=None, cfg_val=1.0, nr_iters_denoise=100, nr_chunks_decode=50):
        self.nr_chunks_decode_strands = nr_chunks_decode
        self.nr_iters_denoise = nr_iters_denoise
        self.cfg_val = cfg_val
        self.path_ckpt_strandcodec = path_ckpt_strandcodec
        self.path_config_difflocks = path_config_difflocks
        self.path_ckpt_difflocks = path_ckpt_difflocks
        self.path_ckpt_rgb2material = path_ckpt_rgb2material
        self.mediapipe_img = Mediapipe()
        self.normalization_dict = DiffLocksDataset.get_normalization_data()
        
        scalp_path = os.path.join(DEFAULT_BODY_DATA_DIR, "scalp.ply")
        if not os.path.exists(scalp_path): scalp_path = "data_loader/difflocks_bodydata/scalp.ply"
        self.scalp_trimesh, self.scalp_mesh_data = DiffLocksDataset.compute_scalp_data(scalp_path)
        self.tbn_space_to_world = tbn_space_to_world_cpu

    def rgb2hair(self, rgb_img, out_path=None):
        if out_path: os.makedirs(out_path, exist_ok=True)
        
        try:
            # ═══════════════════════════════════════════════════════
            # FASE 1: FACE DETECTION (GPU ligero)
            # ═══════════════════════════════════════════════════════
            print("   [1/4] Processing Geometry...")
            frame = (rgb_img.permute(0,2,3,1).squeeze(0)*255).byte().cpu().numpy()
            _, lms = self.mediapipe_img.run(frame)
            if not lms: 
                print("   [ERROR] No face detected")
                return None, None
            
            cropped_face = crop_face(frame, lms, 770)
            rgb_img_gpu = torch.tensor(cropped_face).cuda().permute(2,0,1).unsqueeze(0).float()/255.0
            
            # ═══════════════════════════════════════════════════════
            # FASE 2: DINO FEATURES (GPU)
            # ═══════════════════════════════════════════════════════
            print("   [2/4] Extracting DINO features...")
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', verbose=False).cuda()
            tf = T.Compose([T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
            
            with torch.no_grad(): 
                out = dinov2.forward_features(tf(rgb_img_gpu))
            
            patch = out["x_norm_patchtokens"].clone()
            cls_tok = out["x_norm_clstoken"].clone()
            h = w = int(patch.shape[1]**0.5)
            patch_emb = patch.reshape(patch.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            
            # Guardar copia en CPU ANTES de limpiar
            patch_emb_cpu = patch_emb.cpu().clone()
            cls_tok_cpu = cls_tok.cpu().clone()
            
            # Limpiar DINO completamente
            del dinov2, out, patch
            force_cleanup()
            
            # ═══════════════════════════════════════════════════════
            # FASE 3: DIFFUSION (GPU - la más pesada)
            # ═══════════════════════════════════════════════════════
            print("   [3/4] Running Diffusion (FP32)...")
            
            # Reconstruir en GPU solo lo necesario
            patch_emb_gpu = patch_emb_cpu.cuda()
            cls_tok_gpu = cls_tok_cpu.cuda()
            
            conf = K.config.load_config(self.path_config_difflocks)
            model = K.config.make_denoiser_wrapper(conf)(K.config.make_model(conf).cuda())
            
            ckpt = torch.load(self.path_ckpt_difflocks, map_location='cpu')
            model.inner_model.load_state_dict(ckpt['model_ema'])
            del ckpt
            force_cleanup()
            
            model.eval()
            model.inner_model.condition_dropout_rate = 0.0
            
            extra = {'latents_dict': {"dinov2": {"cls_token": cls_tok_gpu, "final_latent": patch_emb_gpu}}}
            scalp = sample_images_cfg(1, self.cfg_val, [-1., 10000.], model, conf['model'], self.nr_iters_denoise, extra)
            
            # CRÍTICO: Mover resultado a CPU INMEDIATAMENTE
            scalp_cpu = scalp.cpu().clone()
            
            # Limpiar TODA la GPU antes del decoding
            del model, scalp, patch_emb_gpu, cls_tok_gpu, extra
            force_cleanup()
            print("   [INFO] GPU cleared for decoding phase")
            
            # Calcular density en CPU
            density = (scalp_cpu[:,-1:]*(0.5/conf['model']["sigma_data"])+0.5).clamp(0,1)
            density[density<0.02] = 0.0
            
            if density.sum() == 0:
                print("   [ERROR] Empty density map")
                return None, None
            
            # ═══════════════════════════════════════════════════════
            # FASE 4: DECODING (CPU - evita OOM)
            # ═══════════════════════════════════════════════════════
            print("   [4/4] Decoding strands (CPU mode)...")
            
            # Preparar datos en CPU
            norm_dict_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.normalization_dict.items()}
            mesh_data_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.scalp_mesh_data.items()}
            
            # Cargar codec en CPU
            codec = StrandCodec(do_vae=False, decode_type="dir", nr_verts_per_strand=256).cpu()
            codec.load_state_dict(torch.load(self.path_ckpt_strandcodec, map_location='cpu', weights_only=False))
            codec.eval()
            
            # Decodificar en CPU
            strands, _ = sample_strands_from_scalp_with_density(
                scalp_cpu[:,0:-1], density, codec, norm_dict_cpu, 
                mesh_data_cpu, self.tbn_space_to_world.cpu(), self.nr_chunks_decode_strands
            )
            
            del codec
            force_cleanup()
            
            # ═══════════════════════════════════════════════════════
            # FASE 5: MATERIALS (Opcional, GPU si hay espacio)
            # ═══════════════════════════════════════════════════════
            mats = None
            if self.path_ckpt_rgb2material and os.path.exists(self.path_ckpt_rgb2material):
                try:
                    print("   [5/5] Computing materials...")
                    rgb2mat = RGB2MaterialModel(1024, 11, 64).cuda()
                    rgb2mat.load_state_dict(torch.load(self.path_ckpt_rgb2material, map_location='cpu'))
                    rgb2mat.eval()
                    with torch.no_grad(): 
                        mats = rgb2mat({"dinov2_latents": patch_emb_cpu.cuda()})
                    del rgb2mat
                    force_cleanup()
                except Exception as e:
                    print(f"   [WARN] Materials skipped: {e}")
            
            # Guardar resultados
            if out_path and strands is not None:
                np.savez(os.path.join(out_path, "difflocks_output_strands.npz"), positions=strands.cpu().numpy())
                torchvision.utils.save_image(rgb_img_gpu.cpu(), os.path.join(out_path, "rgb.png"))
                if mats:
                    import json
                    with open(os.path.join(out_path, "hair.json"), 'w') as f:
                        json.dump({k: v.tolist() for k, v in mats.items()}, f)
            
            print("   [SUCCESS] Inference complete!")
            return strands, mats
            
        except Exception as e:
            print(f"   [ERROR] {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            force_cleanup()

    def file2hair(self, fpath, out):
        img = cv2.imread(fpath)
        if img is None: raise FileNotFoundError(f"{fpath}")
        rgb = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda().permute(2,0,1).unsqueeze(0).float()/255.
        return self.rgb2hair(rgb, out)
