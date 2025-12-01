import os
import torch
import numpy as np
import k_diffusion as K
import torchvision.transforms as T
from models.strand_codec import StrandCodec
from models.rgb_to_material import RGB2MaterialModel
from utils.diffusion_utils import sample_images_cfg
from data_loader.difflocks_bodydata import crop_face
from utils.strand_util import sample_strands_from_scalp_with_density
from data_loader.dataloader import DiffLocksDataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2

DEFAULT_BODY_DATA_DIR = "data_loader/difflocks_bodydata/body_data"
tbn_space_to_world = torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]).float().cuda()

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
        self.scalp_trimesh, self.scalp_mesh_data = DiffLocksDataset.compute_scalp_data(
            os.path.join(DEFAULT_BODY_DATA_DIR, "scalp.ply"))

    def _clean(self, *objs):
        """Limpieza segura de memoria VRAM"""
        import gc
        # Borramos referencias locales explícitamente
        for o in objs:
            del o
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def rgb2hair(self, rgb_img, out_path=None):
        if out_path: os.makedirs(out_path, exist_ok=True)
        dinov2=model=codec=rgb2mat=None
        try:
            print("   [1/4] Processing Geometry...")
            frame = (rgb_img.permute(0,2,3,1).squeeze(0)*255).byte().cpu().numpy()
            _, lms = self.mediapipe_img.run(frame)
            if not lms: return None, None
            rgb_img = torch.tensor(crop_face(frame, lms, 770)).cuda().permute(2,0,1).unsqueeze(0).float()/255.0
            
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', verbose=False).cuda()
            tf = T.Compose([T.Resize(770), T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
            with torch.no_grad(): out = dinov2.forward_features(tf(rgb_img))
            patch, cls_tok = out["x_norm_patchtokens"].clone(), out["x_norm_clstoken"].clone()
            h = w = int(patch.shape[1]**0.5)
            patch_emb = patch.reshape(patch.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            self._clean(dinov2); dinov2=None

            print("   [2/4] Diffusion (FP32)...")
            conf = K.config.load_config(self.path_config_difflocks)
            model = K.config.make_denoiser_wrapper(conf)(K.config.make_model(conf).cuda())
            ckpt = torch.load(self.path_ckpt_difflocks, map_location='cpu')
            model.inner_model.load_state_dict(ckpt['model_ema'])
            del ckpt; model.eval(); model.inner_model.condition_dropout_rate = 0.0
            
            extra = {'latents_dict': {"dinov2": {"cls_token": cls_tok, "final_latent": patch_emb}}}
            scalp = sample_images_cfg(1, self.cfg_val, [-1., 10000.], model, conf['model'], self.nr_iters_denoise, extra)
            self._clean(model); model=None
            
            density = (scalp[:,-1:]*(0.5/conf['model']["sigma_data"])+0.5).clamp(0,1)
            density[density<0.02] = 0.0

            print("   [3/4] Decoding...")
            codec = StrandCodec(do_vae=False, decode_type="dir", nr_verts_per_strand=256).cuda()
            codec.load_state_dict(torch.load(self.path_ckpt_strandcodec))
            codec.eval()
            strands, _ = sample_strands_from_scalp_with_density(
                scalp[:,0:-1], density, codec, self.normalization_dict, 
                self.scalp_mesh_data, tbn_space_to_world, self.nr_chunks_decode_strands)
            self._clean(codec); codec=None
            
            if self.path_ckpt_rgb2material:
                rgb2mat = RGB2MaterialModel(1024, 11, 64).cuda()
                rgb2mat.load_state_dict(torch.load(self.path_ckpt_rgb2material))
                rgb2mat.eval()
                with torch.no_grad(): mats = rgb2mat({"dinov2_latents": patch_emb})
                self._clean(rgb2mat)

            if out_path and strands is not None:
                np.savez(os.path.join(out_path, "difflocks_output_strands.npz"), positions=strands.cpu().numpy())
            return strands, mats
            
        except Exception as e:
            print(f"   [ERROR] {e}")
            raise
        finally:
            self._clean(dinov2, model, codec, rgb2mat)

    def file2hair(self, fpath, out):
        img = cv2.imread(fpath)
        if img is None: raise FileNotFoundError(f"{fpath}")
        rgb = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda().permute(2,0,1).unsqueeze(0).float()/255.
        return self.rgb2hair(rgb, out)
