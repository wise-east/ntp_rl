#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sync_ntp_rl.sh [remote-path-or-target] [-n] [-d] [-p]

Sync the current local directory to the default target
hjcho@endeavour.usc.edu:/project2/jonmay_1426/hjcho/ntp_rl via rsync over SSH.

Arguments:
  remote-path-or-target  Optional. Either:
    - An absolute remote path (uses default or $REMOTE_HOST), or
    - A full rsync target like user@host:/absolute/path

Options:
  -n           Dry run (show what would change without modifying remote)
  -d           Delete files on remote that no longer exist locally (rsync --delete)
  -p           Pull mode: sync from remote to current local directory
  -h           Show this help

Notes:
  - Set $REMOTE_HOST to override the default host (hjcho@endeavour.usc.edu).
  - Respects .gitignore automatically (rsync filter).
  - Excludes common artifacts like __pycache__, .venv, .DS_Store by default.
USAGE
}

dry_run=false
enable_delete=false
pull_mode=false

# Parse flags that can appear after an optional positional remote-path.
remote_arg=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  remote_arg="$1"
  shift
fi

while getopts ":ndph" opt; do
  case "$opt" in
    n) dry_run=true ;;
    d) enable_delete=true ;;
    p) pull_mode=true ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync not found. Please install rsync and retry." >&2
  exit 1
fi

SRC_DIR="$(pwd)"
SRC_BASENAME="$(basename "$SRC_DIR")"

# Default remote host and path. Host can be overridden via $REMOTE_HOST.
REMOTE_HOST="${REMOTE_HOST:-hjcho@endeavour.usc.edu}"
DEFAULT_REMOTE_DEST="/project2/jonmay_1426/hjcho/ntp_rl"

# Build full rsync target and split for ssh mkdir
if [[ -n "$remote_arg" ]]; then
  if [[ "$remote_arg" == *:* ]]; then
    REMOTE_TARGET="$remote_arg"
    REMOTE_HOST_EXTRACT="${remote_arg%%:*}"
    REMOTE_PATH_EXTRACT="${remote_arg#*:}"
  else
    REMOTE_TARGET="${REMOTE_HOST}:${remote_arg}"
    REMOTE_HOST_EXTRACT="$REMOTE_HOST"
    REMOTE_PATH_EXTRACT="$remote_arg"
  fi
else
  REMOTE_TARGET="${REMOTE_HOST}:${DEFAULT_REMOTE_DEST}"
  REMOTE_HOST_EXTRACT="$REMOTE_HOST"
  REMOTE_PATH_EXTRACT="$DEFAULT_REMOTE_DEST"
fi

direction_label=$([[ "$pull_mode" == true ]] && echo "pull (remote -> local)" || echo "push (local -> remote)")
echo "Direction:     $direction_label"
echo "Local:         $SRC_DIR/"
echo "Remote:        $REMOTE_TARGET/"
echo "Delete:        $([[ "$enable_delete" == true ]] && echo yes || echo no)"
echo "Dry run:       $([[ "$dry_run" == true ]] && echo yes || echo no)"

# For push mode, ensure destination directory exists on remote (expands ~ on remote if present)
if [[ "$pull_mode" != true ]]; then
  echo "Ensuring remote directory exists..."
  ssh "$REMOTE_HOST_EXTRACT" "mkdir -p $REMOTE_PATH_EXTRACT"
fi

rsync_args=(
  -azP
  --human-readable
  --exclude=.DS_Store
  --exclude=**/__pycache__/
  --exclude=.pytest_cache/
  --exclude=.mypy_cache/
  --exclude=.venv/
  --exclude=venv/
  --exclude=.env
  --exclude=.conda/
  --include=requirements-quietstar.txt
  --include=requirements.txt
  --exclude=*.txt
)

if [[ "$enable_delete" == true ]]; then
  rsync_args+=(--delete)
fi

if [[ "$dry_run" == true ]]; then
  rsync_args+=(--dry-run)
fi

set -x
if [[ "$pull_mode" == true ]]; then
  rsync "${rsync_args[@]}" "$REMOTE_TARGET/" "$SRC_DIR/"
else
  rsync "${rsync_args[@]}" "$SRC_DIR/" "$REMOTE_TARGET/"
fi
set +x

echo "Sync complete."


