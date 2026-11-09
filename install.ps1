# lobotomy for Windows: set it up and open it, in one line (PowerShell):
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Vasiniks/lobotomy/main/install.ps1 | iex"
#
# What this does, in order:
#   1. installs uv (https://docs.astral.sh/uv/) into ~\.local\bin if it's missing;
#      uv also fetches Python 3.13 if the PC doesn't have it
#   2. puts lobotomy in ~\lobotomy (a git clone, or a zip without git); run it again to update
#   3. makes a virtual environment there and installs lobotomy with PyTorch: the CUDA
#      build when there's an NVIDIA GPU, the CPU build when there isn't
#   4. downloads the Qwen3.5 0.8B GGUF from Hugging Face (~533 MB) into ~\lobotomy\models
#   5. starts lobotomy and opens it in the browser (Ctrl-C stops it)
#
# Nothing is written outside ~\lobotomy, except uv itself when it was missing.
# Settings: $env:LOBOTOMY_DIR (default ~\lobotomy), $env:LOBOTOMY_NO_START=1 (set up only).

& {
    $ErrorActionPreference = "Stop"
    $Repo = if ($env:LOBOTOMY_REPO) { $env:LOBOTOMY_REPO } else { "https://github.com/Vasiniks/lobotomy" }
    $Dir = if ($env:LOBOTOMY_DIR) { $env:LOBOTOMY_DIR } else { Join-Path $HOME "lobotomy" }

    function Say([string]$msg) { Write-Host $msg -ForegroundColor Cyan }
    # native commands don't throw on failure; stop on a non-zero exit code
    function Run([string]$what, [scriptblock]$cmd) {
        & $cmd
        if ($LASTEXITCODE -ne 0) { throw "lobotomy install: $what failed (exit code $LASTEXITCODE)" }
    }

    if (-not [Environment]::Is64BitOperatingSystem) { throw "lobotomy needs 64-bit Windows." }

    # 1. uv
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        $uvBin = Join-Path $HOME ".local\bin"
        if (-not (Test-Path (Join-Path $uvBin "uv.exe"))) {
            Say "Installing uv (the Python installer lobotomy uses) into ~\.local\bin"
            Run "installing uv" { powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" }
        }
        $env:Path = "$uvBin;$env:Path"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv didn't install; see https://docs.astral.sh/uv/getting-started/installation/"
    }

    # 2. the code
    if (Test-Path (Join-Path $Dir ".git")) {
        Say "Updating $Dir"
        Run "git pull" { git -C $Dir pull --ff-only --quiet }
    } elseif (Test-Path $Dir) {
        if (-not (Test-Path (Join-Path $Dir "pyproject.toml"))) {
            throw "$Dir exists and isn't lobotomy; set `$env:LOBOTOMY_DIR to another folder."
        }
        Say "Using the existing $Dir"
    } elseif (Get-Command git -ErrorAction SilentlyContinue) {
        Say "Downloading lobotomy into $Dir"
        Run "git clone" { git clone --quiet --depth 1 $Repo $Dir }
    } else {
        Say "Downloading lobotomy into $Dir"
        $zip = Join-Path $env:TEMP "lobotomy-main.zip"
        $unpacked = Join-Path $env:TEMP "lobotomy-main"
        Invoke-WebRequest "$Repo/archive/refs/heads/main.zip" -OutFile $zip -UseBasicParsing
        if (Test-Path $unpacked) { Remove-Item $unpacked -Recurse -Force }
        Expand-Archive $zip -DestinationPath $env:TEMP -Force
        Move-Item $unpacked $Dir
        Remove-Item $zip
    }
    Set-Location $Dir

    # 3. the environment
    $hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasNvidia) { Say "Installing lobotomy: PyTorch on your NVIDIA GPU (CUDA)" }
    else { Say "Installing lobotomy: PyTorch on the CPU (no NVIDIA GPU found; this works, but slowly)" }
    $py = Join-Path $Dir ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) { Run "creating the environment" { uv venv --quiet --python 3.13 .venv } }
    Run "installing lobotomy" { uv pip install --quiet --python $py -e ".[torch]" --torch-backend auto }

    # 4. the model
    $lobotomy = Join-Path $Dir ".venv\Scripts\lobotomy.exe"
    Run "downloading the model" { & $lobotomy download }

    Say "Done. Start it any time with:  & `"$lobotomy`" serve --open"

    # 5. start
    if (-not $env:LOBOTOMY_NO_START) {
        Say "Starting lobotomy; the first start converts the weights (~20 s)."
        & $lobotomy serve --open
    }
}
