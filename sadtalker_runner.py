import importlib.util
import os
import runpy
import shutil
import sys
from pathlib import Path

import imageio_ffmpeg


BASE_DIR = Path(__file__).resolve().parent
SADTALKER_SCRIPT = BASE_DIR / "vendor" / "SadTalker" / "inference.py"


def _patch_torchvision_compatibility() -> None:
    if importlib.util.find_spec("torchvision.transforms.functional_tensor") is None:
        import torchvision.transforms._functional_tensor as functional_tensor

        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor


def _patch_ffmpeg_path() -> None:
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = ffmpeg_exe.parent
    ffmpeg_alias = ffmpeg_dir / "ffmpeg.exe"
    if not ffmpeg_alias.exists():
        shutil.copy(ffmpeg_exe, ffmpeg_alias)
    os.environ["PATH"] += os.pathsep + str(ffmpeg_dir)


if __name__ == "__main__":
    _patch_torchvision_compatibility()
    _patch_ffmpeg_path()
    sys.path.insert(0, str(SADTALKER_SCRIPT.parent))
    sys.argv[0] = str(SADTALKER_SCRIPT)
    runpy.run_path(str(SADTALKER_SCRIPT), run_name="__main__")
