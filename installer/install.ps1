# --------------------------------------------------------------------------------------------
# Installation script for LARS-Enterprise/Privion
# Automates 'Installation - Manual' section from the README.md file
# --------------------------------------------------------------------------------------------
# NOTE: This script MUST be run as Administrator to update System PATH and Git System Config.
# --------------------------------------------------------------------------------------------
# Must `Set-ExecutionPolicy RemoteSigned` to run this script!
# Revert with `Set-ExecutionPolicy Restricted` Once Done!
# --------------------------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# --- Admin Check ---
$currentPrincipal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires Administrator privileges to update System Environment Variables." -ForegroundColor Red
    Write-Warning "Please right-click PowerShell and select 'Run as Administrator'." -ForegroundColor Red
    Break
}


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
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
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

if (Test-Path "setup.py") {
    Copy-Item -Path "setup.py" -Destination "$PSScriptRoot\flash-attention\setup.py" -Force
} else {
    Write-Warning "    - [WARNING] setup.py not found in current directory. Using default." -ForegroundColor Yellow
}

# Set MAX_JOBS only for this install step and restore afterwards
# Capturing prior env state first so `finally` is always safe!
$prevMaxJobs = $env:MAX_JOBS
$hadPrevMaxJobs = Test-Path Env:MAX_JOBS

try{
    # Auto-tune MAX_JOBS for Flash-Attention build (safe default: keep 2 threads free)
    $logicalThreads = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    if (-not $logicalThreads -or $logicalThreads -lt 1) {
        $logicalThreads = [Environment]::ProcessorCount
    }

    $reserveThreads = 2
    $maxJobs = [Math]::Max(1, $logicalThreads - $reserveThreads)

    Write-Host "      > Detected $logicalThreads logical threads. Setting MAX_JOBS=$maxJobs for FA2 build..." -ForegroundColor DarkCyan
    $env:MAX_JOBS = "$maxJobs"
    pip install ./flash-attention --no-build-isolation

} catch {
    Write-Host "      > Error determining Logical Thread count for Flash-Attention 2 installation. Proceeding with default (1 CPU core)." -ForegroundColor Yellow
    $env:MAX_JOBS = "1"
    pip install ./flash-attention --no-build-isolation
    
} finally {
    if ($hadPrevMaxJobs) {
        $env:MAX_JOBS = $prevMaxJobs
    } else {
        Remove-Item Env:MAX_JOBS -ErrorAction SilentlyContinue
    }
}

Write-Host "    - [OK] Flash-Attention 2 installed." -ForegroundColor Green


# ---------------------------------------------------------
# Install ASR & TTS dependencies
# ---------------------------------------------------------

Write-Host "    - [5] Installing ASR & TTS dependencies..." -ForegroundColor Yellow

# Install 7-Zip (Needed for FFmpeg extraction) and ESpeak-NG (Needed for TTS)
winget install --id=7zip.7zip -e --accept-source-agreements --accept-package-agreements --disable-interactivity
winget install --id=eSpeak-NG.eSpeak-NG -e --accept-source-agreements --accept-package-agreements --disable-interactivity
Write-Host "    - [OK] 7-Zip & ESpeak-NG installed." -ForegroundColor Green

# Install PIP dependencies, reset versions and ensure hf_xet is uninstalled
pip install -r reqs_speech.txt
pip install "nemo_toolkit[asr]"
pip install numpy==2.2.6
pip install transformers==4.57.3
pip uninstall -y hf_xet
Write-Host "    - [OK] ASR & TTS dependencies installed." -ForegroundColor Green

# --- FFMpeg Installation ---
Write-Host "    - [6] Downloading & Setting up FFmpeg..." -ForegroundColor Yellow

$FFmpegDirName = "ffmpeg-shared"
$FFmpegFinalPath = Join-Path $PSScriptRoot $FFmpegDirName
$FFmpegBinPath = Join-Path $FFmpegFinalPath "bin"

if (Test-Path $FFmpegBinPath) {
    Write-Host "      > FFmpeg already downloaded. Skipping." -ForegroundColor Green
} else {
    # 1. Download v8.0.1
    $FFUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z"
    $FFArchive = Join-Path $PSScriptRoot "ffmpeg_download.7z"

    Write-Host "      > Downloading FFmpeg Full-Shared Release..." -ForegroundColor DarkCyan
    Invoke-WebRequest -Uri $FFUrl -OutFile $FFArchive -UserAgent "PowerShell"

    # 2. Extract using 7-Zip
    # We use the direct path to 7z.exe because the PATH update from winget might not apply to the current session yet
    $7zPath = "$env:ProgramFiles\7-Zip\7z.exe"

    if (-not (Test-Path $7zPath)) {
        throw "7-Zip installation failed or could not be found at $7zPath."
    }

    Write-Host "      > Extracting Archive..." -ForegroundColor DarkCyan
    & $7zPath x $FFArchive -o"$PSScriptRoot" -y | Out-Null

    # 3. Rename folder to standard name
    # The extraction creates a versioned folder (e.g. ffmpeg-7.1-full_build-shared)
    # We find it and rename it to 'ffmpeg-shared' so variables stay constant
    $extractedFolder = Get-ChildItem -Path $PSScriptRoot -Directory -Filter "ffmpeg-*-shared" | Select-Object -First 1

    if ($extractedFolder) {
        Rename-Item -Path $extractedFolder.FullName -NewName $FFmpegDirName
        Remove-Item -Path $FFArchive -Force
        Write-Host "      > [OK] FFmpeg extracted and prepared." -ForegroundColor Green
    } else {
        throw "Could not find extracted FFmpeg folder."
    }

}

# --- Set PATH ---
Write-Host "      > Configuring FFmpeg PATH variable..." -ForegroundColor DarkCyan
if (Test-Path -Path $FFmpegBinPath) {
    # 1. Handle Permanent System PATH (Machine Scope)
    # We must explicitly read the MACHINE path, not the current process path, to avoid polluting System vars with User vars.
    $MachinePath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)

    if ($MachinePath -split ";" -contains $FFmpegBinPath) {
        Write-Host "    - Already in Machine PATH. Skipping." -ForegroundColor Green
    } else {
        $NewEnvPath = "$MachinePath;$FFmpegBinPath"
        [System.Environment]::SetEnvironmentVariable("Path", $NewEnvPath, [System.EnvironmentVariableTarget]::Machine)
        Write-Host "    - [OK] Added FFMpeg to Machine PATH." -ForegroundColor Green
    }

     # 2. Handle Current Session PATH (So the rest of the script/user session works immediately)
     if ($env:PATH -split ";" -notcontains $FFmpegBinPath) {
        $env:PATH += ";$FFmpegBinPath"
        Write-Host "    - [OK] Added to Current Session PATH." -ForegroundColor Green
    }
} else {
    Write-Error "    - [ERROR] FFmpeg not found at $FFmpegBinPath" -ForegroundColor Red
}


Write-Host "    - [7] Copying FFMpeg bin to TorchCodec..." -ForegroundColor Yellow
try {
    # Dynamically get the site-packages path using sysconfig - most reliable way to get the path to the site-packages directory
    $PythonPath = python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    $PythonPath = $PythonPath.Trim()
    $destinationTorchCodecPath = Join-Path $PythonPath "torchcodec"
    $sourceFFMpegPath = Join-Path $FFmpegBinPath "*"

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
    git clone -b v0.0.21 https://github.com/turboderp-org/exllamav3.git
}
pip install ./exllamav3 --no-build-isolation
Write-Host "    - [OK] ExLlamaV3 installed." -ForegroundColor Green

Write-Host "========================================================================================" -ForegroundColor Green
Write-Host "   INSTALLATION COMPLETE! PLEASE FOLLOW FIRST-RUN INSTRUCTIONS IN THE README.MD FILE    " -ForegroundColor Green
Write-Host "========================================================================================" -ForegroundColor Green