import io
import importlib
import gc
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import imageio_ffmpeg
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from sadtalker_service import get_sadtalker_health, render_video_with_sadtalker

cv2 = importlib.import_module("cv2")

BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor" / "Wav2Lip"
MODELS_DIR = BASE_DIR / "models" / "wav2lip"
UPSCALE_MODELS_DIR = BASE_DIR / "models" / "upscale"
WAV2LIP_SCRIPT = VENDOR_DIR / "inference.py"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 25
DEFAULT_UPSCALE_FACTOR = 4
DEFAULT_VIDEO_ENGINE = "wav2lip"
SAFE_MAX_INFERENCE_WIDTH = 1280
SAFE_MAX_INFERENCE_HEIGHT = 720
SAFE_FACE_DET_BATCH = 4
SAFE_WAV2LIP_BATCH = 8
UPSCALE_TILE = 256
UPSCALE_TILE_PAD = 10
UPSCALE_FACE_WEIGHT = 0.5
GFPGAN_MODEL_NAME = "GFPGANv1.4.pth"
GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
REALESRGAN_MODEL_NAME = "realesr-general-x4v3.pth"
REALESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
WAV2LIP_FOLDER_URL = "https://drive.google.com/drive/folders/1I-0dNLfFOSFwrfqjNa-SXuwaURHE5K4k?usp=sharing"
WAV2LIP_FACE_DETECTOR_URL = (
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
)
WAV2LIP_FACE_DETECTOR_PATH = (
    VENDOR_DIR / "face_detection" / "detection" / "sfd" / "s3fd.pth"
)
SUPPORTED_VIDEO_ENGINES = {"wav2lip", "sadtalker"}


def _get_vram_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
    }


def _resolve_checkpoint() -> Path:
    env_override = os.getenv("WAV2LIP_CHECKPOINT")
    if env_override:
        checkpoint = Path(env_override)
        if checkpoint.is_file():
            return checkpoint

    _ensure_wav2lip_assets()

    candidates = [
        MODELS_DIR / "Wav2Lip-SD-NOGAN.pt",
        MODELS_DIR / "Wav2Lip-SD-GAN.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No Wav2Lip checkpoint found. Expected models/wav2lip/Wav2Lip-SD-NOGAN.pt or Wav2Lip-SD-GAN.pt"
    )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    if destination.is_file():
        destination.unlink()

    from torch.hub import download_url_to_file

    download_url_to_file(url, str(destination), progress=True)


def _ensure_wav2lip_assets() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _download_file(WAV2LIP_FACE_DETECTOR_URL, WAV2LIP_FACE_DETECTOR_PATH)

    if any(
        (MODELS_DIR / name).is_file()
        for name in ("Wav2Lip-SD-NOGAN.pt", "Wav2Lip-SD-GAN.pt")
    ):
        return

    bootstrap_dir = BASE_DIR / ".bootstrap" / "wav2lip"
    if bootstrap_dir.exists():
        shutil.rmtree(bootstrap_dir)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            WAV2LIP_FOLDER_URL,
            "-O",
            str(bootstrap_dir),
        ],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        combined = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
        raise RuntimeError(
            "Failed to download Wav2Lip checkpoints into apps/lipsync_server/models/wav2lip. "
            + (combined or "No additional output was produced.")
        )

    found_checkpoint = False
    for filename in ("Wav2Lip-SD-NOGAN.pt", "Wav2Lip-SD-GAN.pt"):
        matches = list(bootstrap_dir.rglob(filename))
        if not matches:
            continue
        destination = MODELS_DIR / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(matches[0]), str(destination))
        found_checkpoint = True

    shutil.rmtree(bootstrap_dir, ignore_errors=True)

    if not found_checkpoint:
        raise RuntimeError(
            "Wav2Lip checkpoint download completed, but the expected checkpoint files were not found."
        )


def _patch_torchvision_compatibility() -> None:
    if importlib.util.find_spec("torchvision.transforms.functional_tensor") is None:
        import torchvision.transforms._functional_tensor as functional_tensor

        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor


def _ensure_upscale_weights() -> tuple[Path, Path]:
    _patch_torchvision_compatibility()

    from basicsr.utils.download_util import load_file_from_url

    UPSCALE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    gfpgan_path = UPSCALE_MODELS_DIR / GFPGAN_MODEL_NAME
    if gfpgan_path.is_file() and gfpgan_path.stat().st_size == 0:
        gfpgan_path.unlink()
    if not gfpgan_path.is_file():
        load_file_from_url(
            GFPGAN_MODEL_URL,
            model_dir=str(UPSCALE_MODELS_DIR),
            file_name=GFPGAN_MODEL_NAME,
        )

    realesrgan_path = UPSCALE_MODELS_DIR / REALESRGAN_MODEL_NAME
    if realesrgan_path.is_file() and realesrgan_path.stat().st_size == 0:
        realesrgan_path.unlink()
    if not realesrgan_path.is_file():
        load_file_from_url(
            REALESRGAN_MODEL_URL,
            model_dir=str(UPSCALE_MODELS_DIR),
            file_name=REALESRGAN_MODEL_NAME,
        )

    return gfpgan_path, realesrgan_path


def _upscale_with_gfpgan(
    image_path: Path, output_path: Path, scale: int
) -> tuple[int, int, int, int]:
    if scale not in {2, 4}:
        raise ValueError("Upscale factor must be 2 or 4.")

    _patch_torchvision_compatibility()

    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Portrait image could not be decoded.")

    original_h, original_w = image.shape[:2]
    if original_h < 8 or original_w < 8:
        raise RuntimeError("Portrait image is too small to upscale reliably.")

    gfpgan_path, realesrgan_path = _ensure_upscale_weights()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    bg_upsampler = None
    face_restorer = None

    try:
        bg_model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        bg_upsampler = RealESRGANer(
            scale=4,
            model_path=str(realesrgan_path),
            model=bg_model,
            tile=UPSCALE_TILE if use_cuda else 0,
            tile_pad=UPSCALE_TILE_PAD,
            pre_pad=0,
            half=use_cuda,
            device=device,
        )
        face_restorer = GFPGANer(
            model_path=str(gfpgan_path),
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
            device=device,
        )

        _, _, restored = face_restorer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=UPSCALE_FACE_WEIGHT,
        )
        if restored is None:
            raise RuntimeError("GFPGAN did not return an upscaled portrait.")

        if scale == 2:
            restored = cv2.resize(
                restored,
                (max(2, original_w * 2), max(2, original_h * 2)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), restored):
            raise RuntimeError("Failed to write the upscaled portrait image.")

        output_h, output_w = restored.shape[:2]
        return original_w, original_h, output_w, output_h
    finally:
        del face_restorer, bg_upsampler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    await upload.close()


def _run_command(command: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = "\n".join(
        part.strip()
        for part in [completed.stdout, completed.stderr]
        if part and part.strip()
    )
    if completed.returncode != 0:
        raise RuntimeError(combined or "Command failed without output.")
    return combined


def _normalize_audio_to_wav(audio_path: Path, output_path: Path) -> None:
    command = [
        FFMPEG_EXE,
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    _run_command(command)


def _prepare_portrait(
    image_path: Path, output_path: Path, width: int, height: int
) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("Portrait image could not be decoded.")

    original_h, original_w = image.shape[:2]
    max_w = max(256, min(width, SAFE_MAX_INFERENCE_WIDTH))
    max_h = max(256, min(height, SAFE_MAX_INFERENCE_HEIGHT))
    scale = min(max_w / original_w, max_h / original_h, 1.0)
    resized_w = max(2, int(round(original_w * scale)))
    resized_h = max(2, int(round(original_h * scale)))

    if scale < 1.0:
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError("Failed to prepare portrait image.")
    return resized_w, resized_h


def _map_upscale_error(error_text: str) -> HTTPException:
    detail = error_text.strip() or "Portrait upscaling failed."
    lower_detail = detail.lower()
    if "out of memory" in lower_detail or "oom" in lower_detail:
        return HTTPException(
            status_code=503,
            detail="GPU memory was exhausted during portrait upscaling. Try a smaller image or use a 2x upscale factor.",
        )
    if "decode" in lower_detail or "decoded" in lower_detail:
        return HTTPException(
            status_code=400,
            detail="The uploaded portrait image could not be decoded.",
        )
    return HTTPException(status_code=500, detail=f"Portrait upscaling failed: {detail}")


def _render_video_with_wav2lip(
    image_path: Path,
    audio_wav_path: Path,
    output_path: Path,
    fps: int,
) -> str:
    checkpoint_path = _resolve_checkpoint()
    temp_dir = VENDOR_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(WAV2LIP_SCRIPT),
        "--checkpoint_path",
        str(checkpoint_path),
        "--face",
        str(image_path),
        "--audio",
        str(audio_wav_path),
        "--outfile",
        str(output_path),
        "--fps",
        str(fps),
        "--pads",
        "0",
        "12",
        "0",
        "0",
        "--face_det_batch_size",
        str(SAFE_FACE_DET_BATCH),
        "--wav2lip_batch_size",
        str(SAFE_WAV2LIP_BATCH),
    ]
    return _run_command(command, cwd=VENDOR_DIR)


def _finalize_video(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
) -> None:
    scale_pad_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    )
    command = [
        FFMPEG_EXE,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        scale_pad_filter,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _run_command(command)


def _map_generation_error(error_text: str, engine: str = "wav2lip") -> HTTPException:
    detail = error_text.strip() or "Video generation failed."
    lower_detail = detail.lower()
    if "face not detected" in lower_detail:
        return HTTPException(
            status_code=400,
            detail=f"{engine.title()} could not detect a face in the portrait. Use a clear front-facing image.",
        )
    if "out of memory" in lower_detail or "oom" in lower_detail:
        return HTTPException(
            status_code=503,
            detail=f"GPU memory was exhausted during {engine} video generation. Try a smaller portrait or lower output resolution.",
        )
    if "sadtalker source not found" in lower_detail or "checkpoint" in lower_detail:
        return HTTPException(status_code=503, detail=detail)
    return HTTPException(status_code=500, detail=f"Video generation failed: {detail}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        checkpoint = _resolve_checkpoint()
        print(f"VoiceForge AI LipSync ready — Wav2Lip checkpoint: {checkpoint}")
    except Exception as exc:
        print(f"VoiceForge AI LipSync startup warning: {exc}")

    upscaler_weights = [
        UPSCALE_MODELS_DIR / GFPGAN_MODEL_NAME,
        UPSCALE_MODELS_DIR / REALESRGAN_MODEL_NAME,
    ]
    missing_weights = [
        str(path.name) for path in upscaler_weights if not path.is_file()
    ]
    if missing_weights:
        print(
            "VoiceForge AI LipSync upscaler ready — weights will download on first upscale request: "
            + ", ".join(missing_weights)
        )
    else:
        print("VoiceForge AI LipSync upscaler ready — local GFPGAN weights detected.")
    yield


app = FastAPI(title="VoiceForge AI — LipSync Server", lifespan=lifespan)


@app.get("/health")
async def health():
    checkpoint = None
    checkpoint_exists = False
    try:
        checkpoint = str(_resolve_checkpoint())
        checkpoint_exists = True
    except Exception:
        checkpoint = None

    return {
        "status": "ok",
        "renderer": "multi-engine",
        "upscaler": "gfpgan+realesrgan",
        "video_engines": sorted(SUPPORTED_VIDEO_ENGINES),
        "checkpoint": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "upscale_default_factor": DEFAULT_UPSCALE_FACTOR,
        "upscale_supported_factors": [2, 4],
        "gfpgan_weights_present": (UPSCALE_MODELS_DIR / GFPGAN_MODEL_NAME).is_file(),
        "realesrgan_weights_present": (
            UPSCALE_MODELS_DIR / REALESRGAN_MODEL_NAME
        ).is_file(),
        "ffmpeg": FFMPEG_EXE,
        "python": sys.executable,
        "default_width": DEFAULT_WIDTH,
        "default_height": DEFAULT_HEIGHT,
        "default_fps": DEFAULT_FPS,
        "default_video_engine": DEFAULT_VIDEO_ENGINE,
        "safe_face_det_batch": SAFE_FACE_DET_BATCH,
        "safe_wav2lip_batch": SAFE_WAV2LIP_BATCH,
        **get_sadtalker_health(),
        **_get_vram_info(),
    }


@app.post("/v1/video/upscale")
async def upscale_portrait(
    image_file: UploadFile = File(...),
    scale: int = Form(DEFAULT_UPSCALE_FACTOR),
):
    if scale not in {2, 4}:
        raise HTTPException(status_code=400, detail="Upscale factor must be 2 or 4.")

    image_ext = Path(image_file.filename or "portrait.png").suffix or ".png"

    with tempfile.TemporaryDirectory(prefix="voiceforge-upscale-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / f"portrait{image_ext}"
        output_path = temp_path / "portrait_upscaled.png"

        await _save_upload(image_file, input_path)

        try:
            original_w, original_h, output_w, output_h = _upscale_with_gfpgan(
                input_path,
                output_path,
                scale=scale,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except HTTPException:
            raise
        except Exception as exc:
            raise _map_upscale_error(str(exc))

        return Response(
            content=output_path.read_bytes(),
            media_type="image/png",
            headers={
                "X-Original-Width": str(original_w),
                "X-Original-Height": str(original_h),
                "X-Upscaled-Width": str(output_w),
                "X-Upscaled-Height": str(output_h),
                "X-Upscale-Factor": str(scale),
            },
        )


@app.post("/v1/video/generate")
async def generate_video(
    portrait_image: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    fps: int = Form(DEFAULT_FPS),
    engine: str = Form(DEFAULT_VIDEO_ENGINE),
):
    if width < 256 or height < 256:
        raise HTTPException(
            status_code=400, detail="Video dimensions must be at least 256x256."
        )
    if fps < 1 or fps > 60:
        raise HTTPException(status_code=400, detail="FPS must be between 1 and 60.")

    selected_engine = (engine or DEFAULT_VIDEO_ENGINE).strip().lower()
    if selected_engine not in SUPPORTED_VIDEO_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video engine '{selected_engine}'. Supported: {sorted(SUPPORTED_VIDEO_ENGINES)}",
        )

    if selected_engine == "wav2lip":
        try:
            _resolve_checkpoint()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    portrait_ext = Path(portrait_image.filename or "portrait.png").suffix or ".png"
    audio_ext = Path(audio_file.filename or "audio.wav").suffix or ".wav"

    with tempfile.TemporaryDirectory(prefix="voiceforge-video-") as temp_dir:
        temp_path = Path(temp_dir)
        portrait_path = temp_path / f"portrait{portrait_ext}"
        audio_path = temp_path / f"audio{audio_ext}"
        prepared_portrait_path = temp_path / "portrait_prepared.png"
        normalized_audio_path = temp_path / "audio_normalized.wav"
        wav2lip_output_path = temp_path / "wav2lip_output.mp4"
        sadtalker_output_path = temp_path / "sadtalker_output.mp4"
        final_output_path = temp_path / "output.mp4"

        await _save_upload(portrait_image, portrait_path)
        await _save_upload(audio_file, audio_path)

        try:
            prepared_w, prepared_h = _prepare_portrait(
                portrait_path,
                prepared_portrait_path,
                width=width,
                height=height,
            )
            print(
                f"[LipSync] Engine={selected_engine} prepared portrait at {prepared_w}x{prepared_h}; target output {width}x{height} @ {fps}fps"
            )
            _normalize_audio_to_wav(audio_path, normalized_audio_path)
            if selected_engine == "wav2lip":
                _render_video_with_wav2lip(
                    prepared_portrait_path,
                    normalized_audio_path,
                    wav2lip_output_path,
                    fps=fps,
                )
                _finalize_video(
                    wav2lip_output_path,
                    final_output_path,
                    width=width,
                    height=height,
                    fps=fps,
                )
            else:
                render_video_with_sadtalker(
                    prepared_portrait_path,
                    normalized_audio_path,
                    sadtalker_output_path,
                    _run_command,
                )
                _finalize_video(
                    sadtalker_output_path,
                    final_output_path,
                    width=width,
                    height=height,
                    fps=fps,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except HTTPException:
            raise
        except Exception as exc:
            raise _map_generation_error(str(exc), engine=selected_engine)

        buffer = io.BytesIO(final_output_path.read_bytes())
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
