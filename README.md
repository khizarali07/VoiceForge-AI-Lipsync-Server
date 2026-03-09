# VoiceForge AI Video Server

FastAPI video service for portrait upscaling and talking-head generation.

## What This App Does

- Serves `GET /health`
- Serves `POST /v1/video/upscale`
- Serves `POST /v1/video/generate`
- Supports two engines:
  - `wav2lip` for faster mouth-focused lip sync
  - `sadtalker` for slower full-head animation

## First-Run Model Download Behavior

- GFPGAN downloads into `apps/lipsync_server/models/upscale`
- Real-ESRGAN downloads into `apps/lipsync_server/models/upscale`
- SadTalker checkpoints download into `apps/lipsync_server/models/sadtalker/checkpoints`
- Wav2Lip checkpoints bootstrap into `apps/lipsync_server/models/wav2lip`
- Wav2Lip face detector downloads into `vendor/Wav2Lip/face_detection/detection/sfd`
- SadTalker GFPGAN support files live in `apps/lipsync_server/gfpgan/weights`

All of these are kept inside the app folder so a clean clone can rebuild the runtime structure locally without relying on unrelated global cache folders.

## Entrypoints

- Main combined server: `main.py` on port 8002
- Optional SadTalker-only debug server: `sadtalker_api.py` on port 8003

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

## Repo Notes

- Do not commit model weights, generated videos, or temporary render output.
- Keep vendored source code, server code, requirements, and documentation only.
- `.gitkeep` files preserve the runtime folder structure in git.