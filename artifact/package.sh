#!/usr/bin/env bash
# package.sh — creates a self-contained artifact distribution directory.
#
# Usage: ./artifact/package.sh [DEST]
#   DEST defaults to ./bonsai-artifact (relative to the repo root)
#
# Output structure:
#   DEST/
#     README.md        <- artifact README
#     bonsai/          <- compiler source
#     artifact/        <- benchmark scripts, apps, scenes, rays

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${1:-${ROOT_DIR}/bonsai-artifact}"

if [[ -e "${DEST}" ]]; then
  echo "Error: destination '${DEST}' already exists. Remove it or pass a different path." >&2
  exit 1
fi

mkdir -p "${DEST}"

echo "Packaging artifact -> ${DEST}"

# Copy compiler source into bonsai/
rsync -a --info=progress2 \
  --exclude='.git' \
  --exclude='.claude' \
  --exclude='.github' \
  --exclude='.vscode' \
  --exclude='.DS_Store' \
  --exclude='.clang-format' \
  --exclude='.clang-tidy' \
  --exclude='.gitignore' \
  --exclude='.gitmodules' \
  --exclude='apps' \
  --exclude='artifact' \
  --exclude='build' \
  --exclude='build-dbg' \
  --exclude='CMakeFiles' \
  --exclude='scripts' \
  --exclude='*.csv' \
  --exclude='*.pdf' \
  --exclude='bonsai-artifact' \
  --exclude='find_non_pareto_layouts.py' \
  "${ROOT_DIR}/" "${DEST}/bonsai/"

# Copy artifact/ contents into artifact/, excluding crud
rsync -a --info=progress2 \
  --exclude='.DS_Store' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='results' \
  --exclude='apps/wos/fcpw/build' \
  "${SCRIPT_DIR}/" "${DEST}/artifact/"

# Symlink stdlib at the root so the compiler can find it from ROOT_DIR
ln -s bonsai/stdlib "${DEST}/stdlib"

# Put the artifact README at the top level
cp "${SCRIPT_DIR}/README.md" "${DEST}/README.md"

echo "Done: ${DEST}"

# Zip the directory
ZIP="${DEST}.zip"
echo "Compressing -> ${ZIP}"
DEST_NAME="$(basename "${DEST}")"
DEST_PARENT="$(dirname "${DEST}")"
(cd "${DEST_PARENT}" && zip -r "${ZIP}" "${DEST_NAME}" \
    -x "*/.DS_Store" \
    -x "*/.venv/*" \
    -x "*/__pycache__/*" \
    -x "*.pyc")
echo "Done: ${ZIP}"
