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

# Imports internos
from models.strand_codec import StrandCodec
from models.rgb_to_material import RGB2MaterialModel
from utils.diffusion_utils import sample_images_cfg
from utils.strand_util import sample_strands_from_scalp_with_density
from data_loader.dataloader import DiffLocksDataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DEFAULT_BODY_DATA_DIR = "data_loader/difflocks_bodydata"
tbn_space_to_world_cpu = torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]).float()
torch.set_num_threads(4)

# --- FUNCIONES AUXILIARES ---
def interpolate_tbn(barys, vertex_idxs, v_tangents, v_bitangents, v_normals):
    nr_positions = barys.shape[0]
    sampled_tangents = v_tangents[vertex_idxs.reshape(-1),:].reshape(nr_positions,3,3)
    weighted_tangents = sampled_tangents * barys.reshape(nr_positions,3,1)
    point_tangents = weighted_tangents.sum(axis=1)
    norm = np.linalg.norm(point_tangents, axis=-1, keepdims=True)
    point_tangents = point_tangents / (norm + 1e-8)

    sampled_normals = v_normals[vertex_idxs.reshape(-1),:].reshape(nr_positions,3,3)
    weighted_normals = sampled_normals * barys.reshape(nr_positions,3,1)
    point_normals = weighted_normals.sum(axis=1)
    norm = np.linalg.norm(point_normals, axis=-1, keepdims=True)
    point_normals = point_normals / (norm + 1e-8)

    point_bitangents = np.cross(point_normals, point_tangents)
    norm = np.linalg.norm(point_bitangents, axis=-1, keepdims=True)
    point_bitangents = point_bitangents / (norm + 1e-8)

    point_tangents = np.cross(point_bitangents, point_normals)
    norm = np.linalg.norm(point_tangents, axis=-1, keepdims=True)
    point_tangents = point_tangents / (norm + 1e-8)
    return point_tangents, point_bitangents, point_normals

def tbn_space_to_world_cpu_safe(root_uv, strands_positions, scalp_mesh_data):
    target_device = strands_positions.device
    target_dtype = torch.float32 
    scalp_index_map = scalp_mesh_data["index_map"]
    scalp_vertex_idxs_map = scalp_mesh_data["vertex_idxs_map"]
    scalp_bary_map = scalp_mesh_data["bary_map"]
    mesh_v_tangents = scalp_mesh_data["v_tangents"]
    mesh_v_bitangents = scalp_mesh_data["v_bitangents"]
    mesh_v_normals = scalp_mesh_data["v_normals"]
    scalp_v = scalp_mesh_data["verts"]
    tex_size = scalp_vertex_idxs_map.shape[0]
    root_uv_np = root_uv.cpu().numpy() if torch.is_tensor(root_uv) else root_uv
    pixel_indices = np.floor(root_uv_np * tex_size).astype(int)
    pixel_indices = np.clip(pixel_indices, 0, tex_size - 1)
    vertex_idxs = scalp_vertex_idxs_map[pixel_indices[:, 0], pixel_indices[:, 1], :]
    barys = scalp_bary_map[pixel_indices[:, 0], pixel_indices[:, 1], :]
    root_tangent, root_bitangent, root_normal = interpolate_tbn(barys, vertex_idxs, mesh_v_tangents, mesh_v_bitangents, mesh_v_normals)
    strands_tbn_np = np.stack((root_tangent, root_bitangent, root_normal), axis=2)
    strands_tbn = torch.as_tensor(strands_tbn_np, device=target_device, dtype=target_dtype)
    indices_tbn = torch.tensor([0, 2, 1], device=target_device, dtype=torch.long)
    strands_tbn = torch.index_select(strands_tbn, 2, indices_tbn)
    strands_tbn[..., 0] = -strands_tbn[..., 0]
    orig_points = torch.matmul(strands_tbn, strands_positions.permute(0, 2, 1)).permute(0, 2, 1)
    nr_positions = vertex_idxs.shape[0]
    sampled_v_np = scalp_v[vertex_idxs.reshape(-1), :].reshape(nr_positions, 3, 3)
    sampled_v = torch.as_tensor(sampled_v_np, device=target_device, dtype=target_dtype)
    barys_tensor = torch.as_tensor(barys, device=target_device, dtype=target_dtype)
    weighted_v = sampled_v * barys_tensor.reshape(nr_positions, 3, 1)
    roots_positions = weighted_v.sum(dim=1)
    strds_points = orig_points + roots_positions[:, None, :]
    return strds_points

def force_cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def crop_face(image, face_landmarks, output_size=770):
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
        asset_path = 'inference/assets/face_landmarker_v2_with_blendshapes.task'
        if not os.path.exists(asset_path): asset_path = os.path.join(os.getcwd(), asset_path)
        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                             output_facial_transformation_matrixes=True, num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
    def run(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        res = self.detector.detect(mp_image)
        return (res.face_blendshapes[0], res.face_landmarks[0]) if res.face_landmarks else (None, None)

class DiffLocksInference():
    def __init__(self, path_ckpt_strandcodec, path_config_difflocks, path_ckpt_difflocks,
                 path_ckpt_rgb2material=None, cfg_val=1.0, nr_iters_denoise=100, nr_chunks_decode=150):
        
        self.nr_chunks_decode_strands = nr_chunks_decode
        self.nr_iters_denoise = nr_iters_denoise
        self.cfg_val = cfg_val
        self.paths = { 'codec': path_ckpt_strandcodec, 'config': path_config_difflocks, 'diff': path_ckpt_difflocks, 'mat': path_ckpt_rgb2material }
        self.mediapipe_img = Mediapipe()
        self.normalization_dict = DiffLocksDataset.get_normalization_data()
        scalp_path = os.path.join(DEFAULT_BODY_DATA_DIR, "scalp.ply")
        if not os.path.exists(scalp_path): scalp_path = "data_loader/difflocks_bodydata/scalp.ply"
        self.scalp_trimesh, self.scalp_mesh_data = DiffLocksDataset.compute_scalp_data(scalp_path)
        self.norm_dict_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.normalization_dict.items()}
        self.mesh_data_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.scalp_mesh_data.items()}
        self.tbn_space_to_world = tbn_space_to_world_cpu_safe

    @torch.inference_mode()
    def rgb2hair(self, rgb_img, out_path=None, cfg_val=None):
        if out_path: os.makedirs(out_path, exist_ok=True)
        
        # --- DEBUG EXTREMO ---
        actual_cfg = cfg_val if cfg_val is not None else self.cfg_val
        print(f"\n{'#'*40}")
        print(f"📢 [DEBUG] INFERENCIA INICIADA")
        print(f"👉 CFG RECIBIDO: {actual_cfg} (Tipo: {type(actual_cfg)})")
        print(f"👉 CFG DEFAULT:  {self.cfg_val}")
        print(f"{'#'*40}\n")
        # ---------------------

        try:
            print("   [1/4] Processing Geometry...")
            frame = (rgb_img.permute(0,2,3,1).squeeze(0)*255).byte().cpu().numpy()
            _, lms = self.mediapipe_img.run(frame)
            if not lms: return None, None
            cropped_face = crop_face(frame, lms, 770)
            del frame
            rgb_img_gpu = torch.tensor(cropped_face).cuda().permute(2,0,1).unsqueeze(0).float()/255.0
            rgb_img_cpu = rgb_img_gpu.cpu().clone()
            
            print("   [2/4] DINO Features...")
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', verbose=False).cuda()
            tf = T.Compose([T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
            out = dinov2.forward_features(tf(rgb_img_gpu))
            patch = out["x_norm_patchtokens"]
            cls_tok = out["x_norm_clstoken"]
            h = w = int(patch.shape[1]**0.5)
            patch_emb = patch.reshape(patch.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            patch_emb_cpu = patch_emb.cpu().clone(); cls_tok_cpu = cls_tok.cpu().clone()
            del dinov2, out, patch, cls_tok, patch_emb, rgb_img_gpu
            force_cleanup()
            
            print("   [3/4] Diffusion...")
            conf = K.config.load_config(self.paths['config'])
            model = K.config.make_denoiser_wrapper(conf)(K.config.make_model(conf).cuda())
            ckpt = torch.load(self.paths['diff'], map_location='cpu', weights_only=False)
            model.inner_model.load_state_dict(ckpt['model_ema'])
            del ckpt; force_cleanup()
            model.eval(); model.inner_model.condition_dropout_rate = 0.0
            
            extra = {'latents_dict': {"dinov2": {"cls_token": cls_tok_cpu.cuda(), "final_latent": patch_emb_cpu.cuda()}}}
            
            # USANDO EL CFG ACTUAL
            scalp = sample_images_cfg(1, actual_cfg, [-1., 10000.], model, conf['model'], self.nr_iters_denoise, extra)
            
            scalp_cpu = scalp.cpu().clone(); sigma_data = conf['model']["sigma_data"]
            del model, scalp, extra, conf; force_cleanup()
            
            density = (scalp_cpu[:,-1:]*(0.5/sigma_data)+0.5).clamp(0,1)
            density[density<0.02] = 0.0
            if density.sum() == 0: return None, None
            
            print("   [4/4] Decoding (CPU Mode)...")
            codec = StrandCodec(do_vae=False, decode_type="dir", nr_verts_per_strand=256).cpu()
            codec.load_state_dict(torch.load(self.paths['codec'], map_location='cpu', weights_only=False))
            codec.eval()
            print(f"   [INFO] Chunks: {self.nr_chunks_decode_strands}")
            strands, _ = sample_strands_from_scalp_with_density(
                scalp_cpu[:,0:-1], density, codec, self.norm_dict_cpu, 
                self.mesh_data_cpu, self.tbn_space_to_world, self.nr_chunks_decode_strands)
            del codec, scalp_cpu, density; force_cleanup()
            
            print("   [5/5] Saving...")
            if out_path and strands is not None:
                positions = strands.cpu().numpy()
                np.savez_compressed(os.path.join(out_path, "difflocks_output_strands.npz"), positions=positions)
                del positions
                torchvision.utils.save_image(rgb_img_cpu, os.path.join(out_path, "rgb.png"))
            
            print("   ✅ DONE!")
            return strands, None

        except Exception as e:
            print(f"   ❌ {e}"); traceback.print_exc(); raise
        finally:
            force_cleanup()

    def file2hair(self, fpath, out, cfg_val=None):
        img = cv2.imread(fpath)
        if img is None: raise FileNotFoundError(f"{fpath}")
        rgb = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda().permute(2,0,1).unsqueeze(0).float()/255.
        return self.rgb2hair(rgb, out, cfg_val=cfg_val)
