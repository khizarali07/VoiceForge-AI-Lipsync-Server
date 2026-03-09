import gc
import shutil
import sys
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
SADTALKER_VENDOR_DIR = BASE_DIR / "vendor" / "SadTalker"
SADTALKER_SCRIPT = SADTALKER_VENDOR_DIR / "inference.py"
SADTALKER_RUNNER = BASE_DIR / "sadtalker_runner.py"
SADTALKER_MODELS_DIR = BASE_DIR / "models" / "sadtalker" / "checkpoints"
UPSCALE_MODELS_DIR = BASE_DIR / "models" / "upscale"
GFPGAN_LOCAL_WEIGHTS_DIR = BASE_DIR / "gfpgan" / "weights"
SADTALKER_RESULT_SIZE = 256
SADTALKER_PREPROCESS = "full"
SADTALKER_BATCH_SIZE = 1
SADTALKER_MODEL_NAME = "SadTalker_V0.0.2_256.safetensors"
SADTALKER_MODEL_URL = (
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
    "SadTalker_V0.0.2_256.safetensors"
)
SADTALKER_MAPPING_NAME = "mapping_00109-model.pth.tar"
SADTALKER_MAPPING_URL = (
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
    "mapping_00109-model.pth.tar"
)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    if destination.is_file():
        destination.unlink()

    from torch.hub import download_url_to_file

    download_url_to_file(url, str(destination), progress=True)


def ensure_sadtalker_assets() -> dict:
    if not SADTALKER_SCRIPT.is_file():
        raise FileNotFoundError(
            f"SadTalker source not found at {SADTALKER_SCRIPT}. Expected vendor/SadTalker/inference.py"
        )

    model_path = SADTALKER_MODELS_DIR / SADTALKER_MODEL_NAME
    mapping_path = SADTALKER_MODELS_DIR / SADTALKER_MAPPING_NAME
    shared_gfpgan_path = UPSCALE_MODELS_DIR / "GFPGANv1.4.pth"
    local_gfpgan_path = GFPGAN_LOCAL_WEIGHTS_DIR / "GFPGANv1.4.pth"

    _download_file(SADTALKER_MODEL_URL, model_path)
    _download_file(SADTALKER_MAPPING_URL, mapping_path)
    GFPGAN_LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if shared_gfpgan_path.is_file() and not local_gfpgan_path.is_file():
        shutil.copy2(shared_gfpgan_path, local_gfpgan_path)

    return {
        "vendor_dir": SADTALKER_VENDOR_DIR,
        "script": SADTALKER_SCRIPT,
        "runner": SADTALKER_RUNNER,
        "checkpoint_dir": SADTALKER_MODELS_DIR,
        "model_path": model_path,
        "mapping_path": mapping_path,
        "local_gfpgan_path": local_gfpgan_path,
    }


def get_sadtalker_health() -> dict:
    model_path = SADTALKER_MODELS_DIR / SADTALKER_MODEL_NAME
    mapping_path = SADTALKER_MODELS_DIR / SADTALKER_MAPPING_NAME
    return {
        "sadtalker_vendor_present": SADTALKER_SCRIPT.is_file(),
        "sadtalker_checkpoint_present": model_path.is_file(),
        "sadtalker_mapping_present": mapping_path.is_file(),
        "sadtalker_size": SADTALKER_RESULT_SIZE,
        "sadtalker_preprocess": SADTALKER_PREPROCESS,
        "sadtalker_batch_size": SADTALKER_BATCH_SIZE,
    }


def render_video_with_sadtalker(
    image_path: Path,
    audio_wav_path: Path,
    output_path: Path,
    run_command,
) -> str:
    assets = ensure_sadtalker_assets()
    result_dir = output_path.parent / "sadtalker_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(assets["runner"]),
        "--driven_audio",
        str(audio_wav_path),
        "--source_image",
        str(image_path),
        "--checkpoint_dir",
        str(assets["checkpoint_dir"]),
        "--result_dir",
        str(result_dir),
        "--still",
        "--preprocess",
        SADTALKER_PREPROCESS,
        "--size",
        str(SADTALKER_RESULT_SIZE),
        "--batch_size",
        str(SADTALKER_BATCH_SIZE),
        "--enhancer",
        "gfpgan",
    ]
    if not torch.cuda.is_available():
        command.append("--cpu")

    try:
        output = run_command(command, cwd=BASE_DIR)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidates = sorted(result_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise RuntimeError("SadTalker completed without producing an MP4 output.")

    candidates[-1].replace(output_path)
    return output
