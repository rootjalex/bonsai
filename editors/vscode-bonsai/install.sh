#!/usr/bin/env bash
# Install the Bonsai syntax extension into VS Code by symlinking this
# directory into every extensions directory VS Code keeps here: the desktop
# one (~/.vscode/extensions) and the Remote-SSH / WSL server one
# (~/.vscode-server/extensions). A symlink rather than a copy, so editing the
# grammar here and reloading the window is the whole development loop.
#
#   editors/vscode-bonsai/install.sh            # install (or refresh) the links
#   editors/vscode-bonsai/install.sh --uninstall
#
# Afterwards reload VS Code (Developer: Reload Window) or restart it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
link_name="bonsai.bonsai-syntax"

uninstall=0
if [[ "${1:-}" == "--uninstall" ]]; then
    uninstall=1
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--uninstall]" >&2
    exit 2
fi

found=0
for dir in "$HOME/.vscode/extensions" "$HOME/.vscode-server/extensions" \
           "$HOME/.vscode-insiders/extensions" "$HOME/.vscode-oss/extensions"; do
    [[ -d "$dir" ]] || continue
    found=1
    target="$dir/$link_name"
    if [[ $uninstall -eq 1 ]]; then
        if [[ -L "$target" ]]; then
            rm "$target"
            echo "removed $target"
        fi
        continue
    fi
    if [[ -e "$target" && ! -L "$target" ]]; then
        echo "$target exists and is not a symlink; not touching it" >&2
        continue
    fi
    ln -sfn "$here" "$target"
    echo "linked $target -> $here"
done

if [[ $found -eq 0 ]]; then
    echo "no VS Code extensions directory found under $HOME" >&2
    echo "symlink or copy $here into your extensions directory by hand" >&2
    exit 1
fi
