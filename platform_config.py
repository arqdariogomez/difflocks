
"""
DiffLocks Platform Configuration Module
Autodetects: Kaggle, Colab, HuggingFace Spaces, Pinokio, Local.
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

PlatformType = Literal['kaggle', 'colab', 'huggingface', 'pinokio', 'local']

@dataclass
class Config:
    platform: PlatformType
    work_dir: Path
    repo_dir: Path
    output_dir: Path
    blender_exe: Path
    has_gpu: bool
    needs_share: bool

    @staticmethod
    def detect() -> 'Config':
        if Path("/kaggle").exists():
            platform = 'kaggle'
            work_dir = Path("/kaggle/working")
            repo_dir = work_dir / "difflocks"
            blender_exe = work_dir / "blender/blender"
            needs_share = True
        elif 'COLAB_GPU' in os.environ or Path("/content").exists():
            platform = 'colab'
            work_dir = Path("/content")
            repo_dir = work_dir / "difflocks"
            blender_exe = work_dir / "blender/blender"
            needs_share = True
        elif 'SPACE_ID' in os.environ: 
            platform = 'huggingface'
            work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = Path("/tmp/blender/blender") 
            needs_share = False
        elif 'PINOKIO_HOME' in os.environ:
            platform = 'pinokio'
            work_dir = Path.cwd()
            repo_dir = work_dir / "app"
            blender_exe = work_dir / "blender" / ("blender.exe" if sys.platform == 'win32' else "blender")
            needs_share = False
        else:
            platform = 'local'
            work_dir = Path.cwd()
            repo_dir = work_dir
            blender_exe = Path("blender/blender") 
            needs_share = False

        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except: pass

        output_dir = work_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        return Config(platform, work_dir, repo_dir, output_dir, blender_exe, has_gpu, needs_share)

cfg = Config.detect()
