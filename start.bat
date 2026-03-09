@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo Installing CUDA-enabled PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
)
pip install -r requirements.txt
python main.py
