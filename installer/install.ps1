# ---------------------------------------------------------
# Installation script for LARS-Enterprise/Privion
# Automates Installation steps 6-11 from the README.md file
# ---------------------------------------------------------

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   STARTING AUTOMATED INSTALLATION & ENVIRONMENT SETUP    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan


# ---------------------------------------------------------
# Install Standard Dependencies & PyTorch
# ---------------------------------------------------------
Write-Host "    - [1] Installing standard dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
Write-Host "    - [OK] Standard dependencies installed." -ForegroundColor Green

Write-Host "    - [2] Installing Butler requirements..." -ForegroundColor Yellow
pip install -r reqs_butler.txt
Write-Host "    - [OK] Butler requirements installed." -ForegroundColor Green

Write-Host "    - [3] Installing PyTorch..." -ForegroundColor Yellow
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
Write-Host "    - [OK] PyTorch installed." -ForegroundColor Green


# ---------------------------------------------------------
# Install Flash-Attention 2
# ---------------------------------------------------------
Write-Host "    - [4] Installing Flash-Attention 2..." -ForegroundColor Yellow

Write-Host "      > Verifying Ninja installation... (ninja --version)..." -ForegroundColor Yellow
try {
    # Run ninja version and silence output to check the Exit Code
    ninja --version | Out-Null

    # In PowerShell, $LastExitCode holds the integer (0, 1, etc.)
    if ($LastExitCode -eq 0) {
        Write-Host "      > Ninja is working correctly (Exit Code 0)." -ForegroundColor Green
    } else {
        throw "Ninja returned non-zero exit code: $LastExitCode"
    }

} catch {
    Write-Host "      > Ninja check failed or returned non-zero code. Reinstalling..." -ForegroundColor Yellow
    pip uninstall -y ninja
    pip install ninja
}

# Disable filename length limit - Git has a limit of 4096 chars for a filename which can lead to "Filename too long" errors when compiling FA2. 
git config --system core.longpaths true
if (-not (Test-Path "flash-attention")) {
    git clone -b v2.7.4.post1 https://github.com/Dao-AILab/flash-attention.git
}

Copy-Item -Path "setup.py" -Destination "$PSScriptRoot\flash-attention\setup.py" -Force
pip install ./flash-attention --no-build-isolation
Write-Host "    - [OK] Flash-Attention 2 installed." -ForegroundColor Green


# ---------------------------------------------------------
# Install ASR & TTS dependencies
# ---------------------------------------------------------

Write-Host "    - [5] Installing ASR & TTS dependencies..." -ForegroundColor Yellow
winget install --id=eSpeak-NG.eSpeak-NG -e --accept-source-agreements --accept-package-agreements
Write-Host "    - [OK] ESpeak-NG installed." -ForegroundColor Green

pip install -r reqs_speech.txt
pip install "nemo_toolkit['asr']"
Write-Host "    - [OK] ASR & TTS dependencies installed." -ForegroundColor Green

# c. Reset NumPy & Transformers to compatible versions
pip install numpy==2.2.6
pip install transformers==4.57.3

Write-Host "    - [6] Setting FFMpeg to PATH..." -ForegroundColor Yellow
$FFMpegPath = "$PSScriptRoot\ffmpeg-8.0-full_build-shared\bin"

if (Test-Path -Path $FFMpegPath) {
    # 1. Handle Permanent System PATH (Machine Scope)
    # We must explicitly read the MACHINE path, not the current process path, to avoid polluting System vars with User vars.
    $MachinePath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)

    if ($MachinePath -split ";" -contains $FFMpegPath) {
        Write-Host "    - Already in Machine PATH. Skipping." -ForegroundColor Green
    } else {
        Write-Host "    - Not found in Machine PATH. Adding permanently..." -ForegroundColor Yellow
        $NewEnvPath = "$MachinePath;$FFMpegPath"
        [System.Environment]::SetEnvironmentVariable("Path", $NewEnvPath, [System.EnvironmentVariableTarget]::Machine)
        Write-Host "    - [OK] Added to FFMpeg System Environment Variables." -ForegroundColor Green
    }

     # 2. Handle Current Session PATH (So the rest of the script/user session works immediately)
     if ($env:PATH -split ";" -notcontains $FFMpegPath) {
        $env:PATH += ";$FFMpegPath"
        Write-Host "    - [OK] Added to Current Session PATH." -ForegroundColor Green
    }
} else {
    Write-Error "    - [ERROR] FFmpeg not found at $FFMpegPath" -ForegroundColor Red
}


Write-Host "    - [7] Copying FFMpeg bin to TorchCodec..." -ForegroundColor Yellow

try {
    # Dynamically get the site-packages path using sysconfig - most reliable way to get the path to the site-packages directory
    $PythonPath = python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    $PythonPath = $PythonPath.Trim()
    $destinationTorchCodecPath = Join-Path $PythonPath "torchcodec"
    $sourceFFMpegPath = "$PSScriptRoot\ffmpeg-8.0-full_build-shared\bin\*"

    # Ensure destination directory exists before copying
    if (Test-Path -Path $destinationTorchCodecPath) {
        Copy-Item -Path $sourceFFMpegPath -Destination $destinationTorchCodecPath -Force -Recurse -ErrorAction Stop
        Write-Host "    - [OK] FFMpeg bin copied to TorchCodec." -ForegroundColor Green
    } else {
        throw "Copy error: The TorchCodec directory $destinationTorchCodecPath could not be found."
    }
} catch {
    Write-Error "    - [ERROR] Failed to copy FFMpeg bin to TorchCodec: $_" -ForegroundColor Red
}


# ---------------------------------------------------------
# Install ExLlamaV2 & ExLlamaV3
# ---------------------------------------------------------

Write-Host "    - [8] Installing ExLlamaV2..." -ForegroundColor Yellow
if (-not (Test-Path "exllamav2")) {
    git clone -b v0.3.2 https://github.com/turboderp-org/exllamav2.git
}
pip install ./exllamav2 --no-build-isolation
Write-Host "    - [OK] ExLlamaV2 installed." -ForegroundColor Green

Write-Host "    - [9] Installing ExLlamaV3..." -ForegroundColor Yellow
if (-not (Test-Path "exllamav3")) {
    git clone -b v0.0.17 https://github.com/turboderp-org/exllamav3.git
}
pip install ./exllamav3 --no-build-isolation
Write-Host "    - [OK] ExLlamaV3 installed." -ForegroundColor Green

Write-Host "========================================================================================" -ForegroundColor Green
Write-Host "   INSTALLATION COMPLETE! PLEASE FOLLOW FIRST-RUN INSTRUCTIONS IN THE README.MD FILE    " -ForegroundColor Green
Write-Host "========================================================================================" -ForegroundColor Green