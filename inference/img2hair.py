#!/usr/bin/env python3


import torch
import cv2
import numpy as np
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import torchvision
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import os
from models.strand_codec import StrandCodec
from models.rgb_to_material import RGB2MaterialModel
import numpy as np
import time
import numpy as np
from models.strand_codec import StrandCodec
from utils.strand_util import sample_strands_from_scalp_with_density
from utils.diffusion_utils import sample_images_cfg

import torch
import torch._dynamo
import torchvision
import sys
import os
from utils.vis_util import img_2_pca
import torchvision.transforms as T
import k_diffusion as K
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from data_loader.dataloader import DEFAULT_BODY_DATA_DIR, DiffLocksDataset
from data_loader.mesh_utils import tbn_space_to_world
VisionRunningMode = mp.tasks.vision.RunningMode
#https://www.reddit.com/r/learnpython/comments/1cxe5ag/need_help_with_mediapipe/
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

torch.autograd.set_grad_enabled(False)



class Mediapipe():
    def __init__(self, mode):
        super(Mediapipe, self).__init__()

        self.mode=mode

        base_options = python.BaseOptions(
            model_asset_path=os.path.join(SCRIPT_DIR,'./assets/face_landmarker.task'),
            delegate=mp.tasks.BaseOptions.Delegate.CPU
            )
        options = vision.FaceLandmarkerOptions(
                                        running_mode=mode,
                                        base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=False,
                                        num_faces=10,
                                        min_face_detection_confidence=0.1,
                                        min_face_presence_confidence=0.1,
                                        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def run(self, rgb_image_numpy):
    
        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_numpy)


        # STEP 4: Detect face landmarks from the input image.
        if self.mode==VisionRunningMode.IMAGE:
            detection_result = self.detector.detect(image)
        else:
            raise Exception("Sorry, not implemented") 
       

        if len (detection_result.face_landmarks) == 0:
            print('No face detected')
            return None, None
        
        # face_landmarks = detection_result.face_landmarks[0]

        # Find the largest face by bounding box area
        largest_face_index = -1
        max_area = 0

        print("number of faces detected", len(detection_result.face_landmarks))
        for i, landmarks in enumerate(detection_result.face_landmarks):
            x_min = min([lm.x for lm in landmarks])
            x_max = max([lm.x for lm in landmarks])
            y_min = min([lm.y for lm in landmarks])
            y_max = max([lm.y for lm in landmarks])

            # Compute bounding box area
            area = (x_max - x_min) * (y_max - y_min)

            if area > max_area:
                max_area = area
                largest_face_index = i

        if largest_face_index == -1:
            print("No valid face detected")
            return None, None
        
        print("largest_face_index",largest_face_index)

        # Get the landmarks of the largest face
        face_landmarks = detection_result.face_landmarks[largest_face_index]


        face_landmarks_numpy = np.zeros((478, 3))

        for i, landmark in enumerate(face_landmarks):
            face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

        # face_landmarks = [(int(lm[0] * img_w), int(lm[1] * img_h)) for lm in face_landmarks]
        fm = []
        for i, landmark in enumerate(face_landmarks):
            fm.append((landmark.x, landmark.y))

        return face_landmarks_numpy, fm



def crop_face(image, face_landmarks, output_size, crop_size_multiplier=2.8):
    img_h, img_w, _ = image.shape


    if face_landmarks is None:
        return cv2.resize(image, (output_size, output_size))
    

    #v4==========
    # Convert normalized landmarks to pixel coordinates
    face_landmarks_px = [(int(lm[0] * img_w), int(lm[1] * img_h)) for lm in face_landmarks]

    # Key face points
    chin = face_landmarks_px[152]  # Chin point
    forehead = face_landmarks_px[10]  # Forehead point
    left_cheek = face_landmarks_px[234]  # Left cheek
    right_cheek = face_landmarks_px[454]  # Right cheek

    # Calculate face bounding box dimensions
    face_width = right_cheek[0] - left_cheek[0]
    face_height = chin[1] - forehead[1]

    # Calculate new face height while maintaining aspect ratio
    crop_size = max(face_width, face_height)  # Ensure square crop around the face

    #make the crop slightly bigger
    # crop_size=int(crop_size*2.8)
    crop_size=int(crop_size*crop_size_multiplier)

    # Calculate crop center
    face_center_x = (left_cheek[0] + right_cheek[0]) // 2
    # face_center_y = (forehead[1] + chin[1]) // 2
    face_center_y = int(forehead[1]*0.4 + chin[1]*0.6)  #not the middle of the face but more closer to the chin than the forehead


    # Crop boundaries in the original image
    crop_x1 = int(face_center_x - crop_size // 2)
    crop_x2 = crop_x1 + crop_size
    crop_y1 = int(face_center_y - crop_size // 2)
    crop_y2 = crop_y1 + crop_size


    #get how much in each direction do we need to pad with zeros
    pad_left=max(0, -crop_x1)
    pad_right=abs(min(0, img_w-crop_x2))
    pad_top=max(0, -crop_y1)
    pad_bottom=abs(min(0, img_h-crop_y2))


    # Extract the region from the original image
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)

    cropped_region = image[crop_y1:crop_y2, crop_x1:crop_x2]


    #pad it with zeros so that the image is square
    padded_image = cv2.copyMakeBorder(
        cropped_region,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Black padding
    )

    # Decide interpolation method based on whether we’re upscaling or downscaling
    h, w = padded_image.shape[:2]
    if h > output_size or w > output_size:
        interpolation = cv2.INTER_AREA  # downscaling
    else:
        interpolation = cv2.INTER_CUBIC  # upscaling

    padded_image = cv2.resize(padded_image, (output_size, output_size), interpolation=interpolation)

    return padded_image





class DiffLocksInference():
    """Optimized Engine: FP32 + Sequential Loading + Guaranteed Cleanup"""
    
    def __init__(self, path_ckpt_strandcodec, path_config_difflocks, path_ckpt_difflocks,
                 path_ckpt_rgb2material=None, cfg_val=1.0, nr_iters_denoise=100, nr_chunks_decode=50):
        super(DiffLocksInference, self).__init__()
        self.nr_chunks_decode_strands = nr_chunks_decode
        self.nr_iters_denoise = nr_iters_denoise
        self.cfg_val = cfg_val
        self.path_ckpt_strandcodec = path_ckpt_strandcodec
        self.path_config_difflocks = path_config_difflocks
        self.path_ckpt_difflocks = path_ckpt_difflocks
        self.path_ckpt_rgb2material = path_ckpt_rgb2material
        self.mediapipe_img = Mediapipe(VisionRunningMode.IMAGE)
        self.normalization_dict = DiffLocksDataset.get_normalization_data()
        
        # Safe load scalp data
        scalp_path = os.path.join(DEFAULT_BODY_DATA_DIR, "scalp.ply")
        if os.path.exists(scalp_path):
            self.scalp_trimesh, self.scalp_mesh_data = DiffLocksDataset.compute_scalp_data(scalp_path)
        else:
            print(f"[WARNING] Scalp data not found at {scalp_path}")

    def _cleanup_vram(self, *objects):
        import gc
        for obj in objects:
            if obj is not None:
                del obj
        gc.collect()
        torch.cuda.empty_cache()

    def rgb2hair(self, rgb_img, out_path=None):
        import gc, torch, numpy as np
        import k_diffusion as K
        import torchvision.transforms as T
        from models.strand_codec import StrandCodec
        from models.rgb_to_material import RGB2MaterialModel
        
        if out_path: 
            os.makedirs(out_path, exist_ok=True)
        
        dinov2 = None; model = None; codec = None; rgb2mat = None
        strands = None; mats = None
        
        try:
            # 1. FACE
            print("   [1/5] Processing facial geometry...")
            frame = (rgb_img.permute(0, 2, 3, 1).squeeze(0) * 255.0).to(torch.uint8).cpu().numpy()
            _, face_landmarks = self.mediapipe_img.run(frame)
            if not face_landmarks: return None, None
            rgb_img = torch.tensor(crop_face(frame, face_landmarks, 770)).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # 2. DINO
            print("   [2/5] Extracting features (DINOv2)...")
            try:
                # Try to load local or hub
                dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', verbose=False).cuda()
                transform = T.Compose([T.Resize(770), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                with torch.no_grad(): out = dinov2.forward_features(transform(rgb_img))
                patch = out["x_norm_patchtokens"].clone()
                cls_tok = out["x_norm_clstoken"].clone()
                h = w = int(patch.shape[1]**0.5)
                patch_emb = patch.reshape(patch.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            finally:
                self._cleanup_vram(dinov2); dinov2 = None
            
            # 3. DIFFUSION
            print("   [3/5] Generating texture (FP32)...")
            extra_args = {'latents_dict': {"dinov2": {"cls_token": cls_tok, "final_latent": patch_emb}}}
            try:
                conf = K.config.load_config(self.path_config_difflocks)
                model = K.config.make_denoiser_wrapper(conf)(K.config.make_model(conf).cuda())
                ckpt = torch.load(self.path_ckpt_difflocks, map_location='cpu')
                model.inner_model.load_state_dict(ckpt['model_ema'])
                del ckpt
                model.eval()
                model.inner_model.condition_dropout_rate = 0.0
                scalp_texture = sample_images_cfg(1, self.cfg_val, [-1.0, 10000.0], model, conf['model'], self.nr_iters_denoise, extra_args)
            finally:
                self._cleanup_vram(model); model = None
                
            density = (scalp_texture[:, -1:, :, :] * (0.5 / conf['model']["sigma_data"]) + 0.5).clamp(0, 1)
            density[density < 0.02] = 0.0

            # 4. CODEC
            print("   [4/5] Decoding strands...")
            try:
                codec = StrandCodec(do_vae=False, decode_type="dir", nr_verts_per_strand=256).cuda()
                codec.load_state_dict(torch.load(self.path_ckpt_strandcodec))
                codec.eval()
                strands, _ = sample_strands_from_scalp_with_density(
                    scalp_texture[:, 0:-1, :, :], density, codec, 
                    self.normalization_dict, self.scalp_mesh_data, 
                    tbn_space_to_world, self.nr_chunks_decode_strands
                )
            finally:
                self._cleanup_vram(codec); codec = None
                
            # 5. MATS
            print("   [5/5] Materials...")
            if self.path_ckpt_rgb2material:
                try:
                    rgb2mat = RGB2MaterialModel(1024, 11, 64).cuda()
                    rgb2mat.load_state_dict(torch.load(self.path_ckpt_rgb2material))
                    rgb2mat.eval()
                    with torch.no_grad(): mats = rgb2mat({"dinov2_latents": patch_emb})
                finally:
                    self._cleanup_vram(rgb2mat); rgb2mat = None

            if out_path and strands is not None:
                np.savez(os.path.join(out_path, "difflocks_output_strands.npz"), positions=strands.cpu().numpy())
                torchvision.utils.save_image(rgb_img, os.path.join(out_path, "rgb.png"))
            
            return strands, mats
            
        except Exception as e:
            print(f"[ERROR INFERENCE] {e}")
            raise e
        finally:
            self._cleanup_vram(dinov2, model, codec, rgb2mat)

    def file2hair(self, file_path, out_path):
        import cv2
        img = cv2.imread(file_path)
        if img is None: raise FileNotFoundError(f"Cannot read: {file_path}")
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = torch.tensor(frame).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return self.rgb2hair(rgb, out_path)
