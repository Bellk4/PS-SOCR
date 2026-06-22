@echo off
setlocal

rem -----------------------------------------------------------------------------
rem GLM-OCR local server launcher (Windows, venv)
rem - Creates/uses .venv next to this script
rem - Installs runtime deps (including transformers dev build)
rem - Starts FastAPI on configured host/port
rem -----------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if /I "%RUNBAT_LOGGED%"=="1" goto :main

set "RUNBAT_LOGGED=1"
set "LOG_DIR=%SCRIPT_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "TS="
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd_HHmmss')"') do set "TS=%%I"
if "%TS%"=="" set "TS=latest"

set "LOG_FILE=%LOG_DIR%\run_%TS%.log"
echo [*] Logging to "%LOG_FILE%"
call "%~f0" %* >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

echo [*] ----- begin log -----
type "%LOG_FILE%"
echo [*] ----- end log -----

echo [*] Exit code: %EXIT_CODE%
echo [*] Log saved: "%LOG_FILE%"
if not "%EXIT_CODE%"=="0" (
    echo [*] Press any key to close...
    pause >nul
)
exit /b %EXIT_CODE%

:main

set "VENV_DIR=%SCRIPT_DIR%\.venv"
set "MODEL_CACHE_DIR=%SCRIPT_DIR%\models\hf_cache"
set "ENV_FILE=%SCRIPT_DIR%\.env"

cd /d "%SCRIPT_DIR%"

if exist "%ENV_FILE%" (
    echo [+] Loading .env from "%ENV_FILE%"
    for /f "usebackq eol=# tokens=1* delims==" %%A in ("%ENV_FILE%") do (
        if not "%%A"=="" (
            set "%%A=%%B"
        )
    )
)

if not exist "%MODEL_CACHE_DIR%" (
    mkdir "%MODEL_CACHE_DIR%"
)

set "HF_HOME=%SCRIPT_DIR%\models\hf_home"
set "HF_HUB_CACHE=%MODEL_CACHE_DIR%"
set "TRANSFORMERS_CACHE=%MODEL_CACHE_DIR%"
set "GLM_MODEL_CACHE=%MODEL_CACHE_DIR%"
if "%TORCH_CHANNEL%"=="" (
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        set "TORCH_CHANNEL=cpu"
    ) else (
        set "TORCH_CHANNEL=cu126"
    )
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [+] Creating virtual environment at "%VENV_DIR%" ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [!] Failed to create virtual environment. Ensure Python 3.10+ is installed and on PATH.
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [!] Failed to activate virtual environment.
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if (sys.version_info.major==3 and 10 <= sys.version_info.minor <= 12) else 1)"
if errorlevel 1 (
    echo [!] Unsupported Python version in this venv.
    python --version
    echo [!] Use Python 3.10-3.12 x64 on Windows Server.
    echo [!] Delete .venv and run again after switching Python.
    exit /b 1
)

echo [+] Installing/ensuring dependencies...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [!] Failed to upgrade pip.
    exit /b 1
)

echo [+] Installing PyTorch (%TORCH_CHANNEL%)...
if /I "%TORCH_CHANNEL%"=="cpu" (
    python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch torchvision
    if errorlevel 1 (
        echo [!] Failed to install PyTorch CPU wheels.
        exit /b 1
    )
) else (
    python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/%TORCH_CHANNEL% torch torchvision
    if errorlevel 1 (
        echo [!] Failed to install PyTorch with TORCH_CHANNEL=%TORCH_CHANNEL%. Falling back to CPU...
        set "TORCH_CHANNEL=cpu"
        python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch torchvision
        if errorlevel 1 (
            echo [!] Failed to install PyTorch CPU fallback.
            exit /b 1
        )
    )
)
python -c "import torch; print('[torch]', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())"
if errorlevel 1 (
    echo [!] PyTorch import check failed after installation.
    echo [!] This is often server-environment dependent (DLL init failure: WinError 1114).
    echo [!] Check these items:
    echo [!] 1) Install Microsoft Visual C++ Redistributable 2015-2022 (x64).
    echo [!] 2) Ensure 64-bit Python 3.10-3.12 is used for this project.
    echo [!] 3) If CPU is old, try an older PyTorch version compatible with the server.
    exit /b 1
)

echo [+] Installing FastAPI and image/PDF dependencies...
python -m pip install fastapi uvicorn websockets wsproto python-multipart pillow pypdfium2 accelerate python-dotenv auth0-server-python httpx
if errorlevel 1 (
    echo [!] Failed to install FastAPI/runtime dependencies.
    exit /b 1
)

echo [+] Installing optional layout dependencies (PaddleOCR)...
python -m pip install --upgrade paddlepaddle
if errorlevel 1 (
    echo [!] paddlepaddle install failed. Layout OCR will use fallback mode.
)
python -m pip install --upgrade paddleocr
if errorlevel 1 (
    echo [!] paddleocr install failed. Layout OCR will use fallback mode.
)

echo [+] Installing transformers (development build)...
python -m pip install git+https://github.com/huggingface/transformers.git
if errorlevel 1 (
    echo [!] Failed to install transformers from GitHub. Check internet/proxy/git settings.
    exit /b 1
)

rem Configure host/port. Override by setting HOST/PORT before running.
if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"

if /I "%AUTH_DISABLED%"=="1" (
    echo [+] Authentication is disabled (AUTH_DISABLED=1)
) else (
    echo [+] Authentication is enabled
)

echo [+] Starting server at http://%HOST%:%PORT%
cd /d "%SCRIPT_DIR%"
python -m uvicorn app.main:app --host "%HOST%" --port "%PORT%"
if errorlevel 1 (
    echo [!] Server start failed. See traceback above.
    exit /b 1
)

endlocal
