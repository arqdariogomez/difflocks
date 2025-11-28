import os
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class Platform(Enum):
    KAGGLE = "kaggle"
    COLAB = "colab"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

def detect_platform() -> Platform:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"): return Platform.KAGGLE
    if os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules: return Platform.COLAB
    if os.environ.get("SPACE_ID"): return Platform.HUGGINGFACE
    return Platform.LOCAL

PLATFORM = detect_platform()

@dataclass
class PlatformPaths:
    base_dir: Path
    models_dir: Path
    output_dir: Path

    @classmethod
    def get(cls):
        p = detect_platform()
        if p == Platform.KAGGLE:
            return cls(
                base_dir=Path("/kaggle/working/difflocks"),
                models_dir=Path("/kaggle/input/difflocks/checkpoints"),
                output_dir=Path("/kaggle/working/outputs")
            )
        elif p == Platform.COLAB:
            return cls(
                base_dir=Path("/content/difflocks"),
                models_dir=Path("/content/models"), # User must link/download here
                output_dir=Path("/content/outputs")
            )
        else: # Local/HF
            return cls(
                base_dir=Path.cwd(),
                models_dir=Path.cwd() / "checkpoints",
                output_dir=Path.cwd() / "outputs"
            )

PATHS = PlatformPaths.get()
