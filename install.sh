#!/usr/bin/env bash
# lobotomy: set it up and open it, in one line (macOS and Linux):
#
#   curl -fsSL https://raw.githubusercontent.com/Vasiniks/lobotomy/main/install.sh | bash
#
# (Windows: see install.ps1.) What this does, in order:
#   1. picks the engine: MLX on Apple Silicon Macs; PyTorch elsewhere, with the
#      CUDA build when there's an NVIDIA GPU and the CPU build when there isn't
#   2. installs uv (https://docs.astral.sh/uv/) into ~/.local/bin if it's missing;
#      uv also fetches Python 3.13 if the machine doesn't have it
#   3. puts lobotomy in ~/lobotomy (a git clone, or a tarball without git);
#      run it again to update
#   4. makes a virtual environment there and installs lobotomy into it
#   5. downloads the Qwen3.5 0.8B GGUF from Hugging Face (~533 MB) into ~/lobotomy/models
#   6. starts lobotomy and opens it in the browser (Ctrl-C stops it)
#
# It also symlinks `lobotomy` and `lobe` into ~/.local/bin so `lobe run` works
# from any shell. Nothing else is written outside ~/lobotomy, except uv itself
# when it was missing. Settings: LOBOTOMY_DIR (default ~/lobotomy),
# LOBOTOMY_NO_START=1 (set up only).
set -euo pipefail

REPO="${LOBOTOMY_REPO:-https://github.com/Vasiniks/lobotomy}"
DIR="${LOBOTOMY_DIR:-$HOME/lobotomy}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf 'lobotomy install: %s\n' "$*" >&2; exit 1; }

# 1. the engine
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS/$ARCH" in
  Darwin/arm64) EXTRA="mlx"; TORCH_FLAG=(); ENGINE="MLX (Apple Silicon)" ;;
  Darwin/*) EXTRA="torch"; TORCH_FLAG=(--torch-backend cpu); ENGINE="PyTorch on the CPU (slow on an Intel Mac)" ;;
  Linux/*)
    EXTRA="torch"; TORCH_FLAG=(--torch-backend auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      ENGINE="PyTorch on your NVIDIA GPU (CUDA)"
    else
      ENGINE="PyTorch on the CPU (no NVIDIA GPU found; this works, but slowly)"
    fi ;;
  *) die "this script is for macOS and Linux; on Windows use install.ps1 (see the README)." ;;
esac

# 2. uv
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    say "Installing uv (the Python installer lobotomy uses) into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
command -v uv >/dev/null 2>&1 || die "uv didn't install; see https://docs.astral.sh/uv/getting-started/installation/"

# 3. the code
have_git() {
  # on macOS without the developer tools, /usr/bin/git only offers to install them
  if [ "$OS" = "Darwin" ]; then xcode-select -p >/dev/null 2>&1 && command -v git >/dev/null 2>&1
  else command -v git >/dev/null 2>&1; fi
}
if [ -d "$DIR/.git" ]; then
  say "Updating $DIR"
  git -C "$DIR" pull --ff-only --quiet
elif [ -e "$DIR" ]; then
  [ -f "$DIR/pyproject.toml" ] || die "$DIR exists and isn't lobotomy; set LOBOTOMY_DIR to another folder."
  say "Using the existing $DIR"
elif have_git; then
  say "Downloading lobotomy into $DIR"
  git clone --quiet --depth 1 "$REPO" "$DIR"
else
  say "Downloading lobotomy into $DIR"
  mkdir -p "$DIR"
  curl -fsSL "$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$DIR" --strip-components 1
fi
cd "$DIR"

# 4. the environment
say "Installing lobotomy: $ENGINE"
[ -x .venv/bin/python ] || uv venv --quiet --python 3.13 .venv
uv pip install --quiet --python .venv/bin/python -e ".[$EXTRA]" ${TORCH_FLAG[@]+"${TORCH_FLAG[@]}"}

# 5. the model
scripts/download_model.sh

# 6. put `lobotomy`/`lobe` on PATH so `lobe run` works in zsh
mkdir -p "$HOME/.local/bin"
for bin in lobotomy lobe; do
  if [ -x "$DIR/.venv/bin/$bin" ]; then
    ln -sf "$DIR/.venv/bin/$bin" "$HOME/.local/bin/$bin"
  fi
done
if ! echo "$PATH" | tr ':' '\n' | grep -q "$HOME/.local/bin"; then
  say "Added ~/.local/bin to PATH — restart your shell or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

say "Done. Start it any time with:  lobe run  (or $DIR/.venv/bin/lobotomy run)"

# 7. start
if [ -z "${LOBOTOMY_NO_START:-}" ]; then
  say "Starting lobotomy; the first start converts the weights (~20 s)."
  exec "$HOME/.local/bin/lobotomy" run 2>/dev/null || exec .venv/bin/lobotomy run
fi
