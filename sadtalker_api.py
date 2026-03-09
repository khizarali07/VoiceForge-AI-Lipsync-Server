import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from main import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    _finalize_video,
    _map_generation_error,
    _normalize_audio_to_wav,
    _prepare_portrait,
    _run_command,
    _save_upload,
)
from sadtalker_service import get_sadtalker_health, render_video_with_sadtalker


app = FastAPI(title="VoiceForge AI — SadTalker Server")


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "sadtalker", **get_sadtalker_health()}


@app.post("/v1/video/generate")
async def generate_video(
    portrait_image: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
    fps: int = Form(DEFAULT_FPS),
):
    if width < 256 or height < 256:
        raise HTTPException(
            status_code=400, detail="Video dimensions must be at least 256x256."
        )
    if fps < 1 or fps > 60:
        raise HTTPException(status_code=400, detail="FPS must be between 1 and 60.")

    portrait_ext = Path(portrait_image.filename or "portrait.png").suffix or ".png"
    audio_ext = Path(audio_file.filename or "audio.wav").suffix or ".wav"

    with tempfile.TemporaryDirectory(prefix="voiceforge-sadtalker-") as temp_dir:
        temp_path = Path(temp_dir)
        portrait_path = temp_path / f"portrait{portrait_ext}"
        audio_path = temp_path / f"audio{audio_ext}"
        prepared_portrait_path = temp_path / "portrait_prepared.png"
        normalized_audio_path = temp_path / "audio_normalized.wav"
        raw_output_path = temp_path / "sadtalker_raw.mp4"
        final_output_path = temp_path / "output.mp4"

        await _save_upload(portrait_image, portrait_path)
        await _save_upload(audio_file, audio_path)

        try:
            _prepare_portrait(
                portrait_path,
                prepared_portrait_path,
                width=width,
                height=height,
            )
            _normalize_audio_to_wav(audio_path, normalized_audio_path)
            render_video_with_sadtalker(
                prepared_portrait_path,
                normalized_audio_path,
                raw_output_path,
                _run_command,
            )
            _finalize_video(
                raw_output_path,
                final_output_path,
                width=width,
                height=height,
                fps=fps,
            )
        except Exception as exc:
            raise _map_generation_error(str(exc), engine="sadtalker")

        buffer = io.BytesIO(final_output_path.read_bytes())
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
