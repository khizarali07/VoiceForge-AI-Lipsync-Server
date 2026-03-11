# VoiceForge AI Video Server

FastAPI video generation microservice engineered explicitly for precise portrait face-upscaling and deep talking-head animation arrays.

## What This App Does

- Exposes standard operational bounds evaluating `GET /health`.
- Processes deep super-resolution tasks `POST /v1/video/upscale` using embedded implementations of GFPGAN & Real-ESRGAN natively boosting image fidelity up to 4x cleanly.
- Processes audio-visual compositing tasks via `POST /v1/video/generate` mapping distinct animation capabilities flexibly dependent on parameter flags.
- Integrates multiple machine learning solutions concurrently:
  - `wav2lip`: Highly optimized native mouth-focused engine execution offering speed-critical rendering.
  - `sadtalker`: Holistic full-head animation algorithm delivering lifelike temporal motion sequences for realistic generation capabilities.

## Core Dependencies

- `fastapi`, `uvicorn`, `python-multipart`
- `torch`, `torchvision`, `torchaudio`
- `opencv-python==4.8.1.78`, `imageio`, `imageio-ffmpeg`, `scikit-image`
- Audio processing: `numpy`, `scipy`, `librosa`, `soundfile`, `resampy`, `pydub`
- ML utilities: `face-alignment`, `av`, `facexlib`, `basicsr`, `gfpgan`, `realesrgan`, `kornia`, `safetensors`, `numba`, `yacs`, `tqdm`

## First-Run Model Download Behavior

- Upscaling structures explicitly populate internally: GFPGAN and Real-ESRGAN into `apps/lipsync_server/models/upscale`.
- SadTalker checkpoints gracefully load to `apps/lipsync_server/models/sadtalker/checkpoints`.
- Wav2Lip model bootstrapping anchors into `apps/lipsync_server/models/wav2lip`.
- The native facial identifier algorithm for Wav2Lip safely registers to `vendor/Wav2Lip/face_detection/detection/sfd`.
- SadTalker GFPGAN structural payloads load successfully within `apps/lipsync_server/gfpgan/weights`.

These localized tracking mechanisms eliminate systemic cache corruptions assuring absolute stability over sequential runs gracefully supporting local deployment execution paradigms.

## Entrypoints

- Native unified server execution context runs successfully utilizing `main.py` directly bound to port `8002`.
- Direct manual API isolation supports local SadTalker rendering instances targeting specifically port `8003`.

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

- Temporary render operations (`.mp4`, `.wav`) completely ignored systematically.
- Internal structural code, logic hooks, API definitions, definitions are preserved accurately.
- Avoid polluting tree with ML checkpoints dynamically.