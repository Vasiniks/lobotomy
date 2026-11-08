#!/usr/bin/env bash
# lobotomy: set it up and open it, in one line:
#
#   curl -fsSL https://raw.githubusercontent.com/Vasiniks/lobotomy/main/install.sh | bash
#
# What this does, in order:
#   1. checks this is an Apple Silicon Mac (lobotomy runs on MLX's Metal kernels)
#   2. installs uv (https://docs.astral.sh/uv/) into ~/.local/bin if it's missing;
#      uv also fetches Python 3.13 if the Mac doesn't have it
#   3. puts lobotomy in ~/lobotomy (a git clone, or a tarball without git);
#      run it again to update
#   4. makes a virtual environment there and installs lobotomy into it
#   5. downloads the Qwen3.5 0.8B GGUF from Hugging Face (~533 MB) into ~/lobotomy/models
#   6. starts lobotomy and opens it in the browser (Ctrl-C stops it)
#
# Nothing is written outside ~/lobotomy, except uv itself when it was missing.
# Settings: LOBOTOMY_DIR (default ~/lobotomy), LOBOTOMY_NO_START=1 (set up only).
set -euo pipefail

REPO="${LOBOTOMY_REPO:-https://github.com/Vasiniks/lobotomy}"
DIR="${LOBOTOMY_DIR:-$HOME/lobotomy}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf 'lobotomy install: %s\n' "$*" >&2; exit 1; }

# 1. Apple Silicon only
[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] \
  || die "lobotomy needs an Apple Silicon Mac (M1 or later); this is $(uname -s) $(uname -m)."

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
if [ -d "$DIR/.git" ]; then
  say "Updating $DIR"
  git -C "$DIR" pull --ff-only --quiet
elif [ -e "$DIR" ]; then
  [ -f "$DIR/pyproject.toml" ] || die "$DIR exists and isn't lobotomy; set LOBOTOMY_DIR to another folder."
  say "Using the existing $DIR"
elif xcode-select -p >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
  # (without the developer tools, /usr/bin/git only offers to install them)
  say "Downloading lobotomy into $DIR"
  git clone --quiet --depth 1 "$REPO" "$DIR"
else
  say "Downloading lobotomy into $DIR"
  mkdir -p "$DIR"
  curl -fsSL "$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$DIR" --strip-components 1
fi
cd "$DIR"

# 4. the environment
say "Installing lobotomy (Python 3.13, MLX)"
[ -x .venv/bin/python ] || uv venv --quiet --python 3.13 .venv
uv pip install --quiet --python .venv/bin/python -e ".[mlx]"

# 5. the model
scripts/download_model.sh

say "Done. Start it any time with:  $DIR/.venv/bin/lobotomy serve --open"

# 6. start
if [ -z "${LOBOTOMY_NO_START:-}" ]; then
  say "Starting lobotomy; the first start converts the weights (~20 s)."
  exec .venv/bin/lobotomy serve --open
fi
