#!/usr/bin/env bash
#
# Switch between Lean 4.15.0 (default) and 4.9.0-rc2 (DeepSeek-Prover compatible).
# Preserves Mathlib and project build caches per version so you don't rebuild from scratch.
#
# Usage:
#   ./scripts/switch_lean_version.sh 4.9.0-rc2   # switch to DeepSeek version
#   ./scripts/switch_lean_version.sh 4.15.0      # switch back to current
#   ./scripts/switch_lean_version.sh             # show current version and usage
#
# Run from the invpro project root.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEAN_ENVS="$PROJECT_ROOT/lean_envs"

VERSION_490="4.9.0-rc2"
VERSION_415="4.15.0"
SUPPORTED_VERSIONS=("$VERSION_490" "$VERSION_415")

cd "$PROJECT_ROOT"

current_version() {
    if [[ -f lean-toolchain ]]; then
        # Parse "leanprover/lean4:v4.15.0" -> 4.15.0, "leanprover/lean4:v4.9.0-rc2" -> 4.9.0-rc2
        local raw
        raw=$(sed -n 's/.*:v\([0-9][0-9.]*[-a-z0-9]*\).*/\1/p' lean-toolchain 2>/dev/null)
        if [[ -n "$raw" ]]; then
            echo "$raw"
            return
        fi
    fi
    echo "unknown"
}

switch_to() {
    local target="$1"
    local env_dir="$LEAN_ENVS/$target"

    if [[ ! -d "$env_dir" ]]; then
        echo "Error: No config for version $target in $env_dir"
        echo "Supported versions: ${SUPPORTED_VERSIONS[*]}"
        exit 1
    fi

    local current
    current=$(current_version)
    echo "Current Lean version: $current"
    echo "Switching to: $target"

    # 1. Preserve current .lake cache by moving to version-specific directory
    if [[ -d .lake ]] && [[ "$current" != "unknown" ]]; then
        local lake_backup=".lake.$current"
        if [[ -d "$lake_backup" ]]; then
            echo "Removing old backup $lake_backup (will be replaced)"
            rm -rf "$lake_backup"
        fi
        echo "Preserving build cache: .lake -> $lake_backup"
        mv .lake "$lake_backup"
    elif [[ -d .lake ]]; then
        # Current version unknown; still back up to avoid mixing caches
        local lake_backup=".lake.bak.$$"
        echo "Preserving .lake as $lake_backup (version unknown)"
        mv .lake "$lake_backup"
    fi

    # 2. Restore or create .lake for target version
    local lake_restore=".lake.$target"
    if [[ -d "$lake_restore" ]]; then
        echo "Restoring build cache: $lake_restore -> .lake"
        mv "$lake_restore" .lake
    else
        echo "No cached build for $target; will fetch and build"
        mkdir -p .lake
    fi

    # 3. Install config files
    cp "$env_dir/lean-toolchain" lean-toolchain
    cp "$env_dir/lakefile.lean" lakefile.lean
    if [[ -f "$env_dir/Rewrites.lean" ]]; then
        cp "$env_dir/Rewrites.lean" invpro/lean/Rewrites.lean
        echo "Installed Rewrites.lean for $target"
    fi
    echo "Installed lean-toolchain and lakefile.lean for $target"

    # 4. Update manifest, fetch Mathlib cache and build
    echo "Updating mathlib dependency..."
    lake update mathlib
    echo "Fetching Mathlib cache..."
    lake exe cache get || true
    echo "Building project..."
    lake build

    echo ""
    echo "Switched to Lean $target. Build cache saved in .lake"
    if [[ "$target" == "$VERSION_490" ]]; then
        echo "To switch back: $0 $VERSION_415"
    else
        echo "To switch back: $0 $VERSION_490"
    fi
}

usage() {
    echo "Usage: $0 [4.9.0-rc2 | 4.15.0]"
    echo ""
    echo " 4.9.0-rc2  - Lean + Mathlib for DeepSeek-Prover compatibility"
    echo " 4.15.0     - Current Lean + Mathlib (default)"
    echo ""
    echo "Current version: $(current_version)"
    echo "Build caches are preserved in .lake.<version> when switching."
}

case "${1:-}" in
    4.9.0-rc2)
        switch_to "4.9.0-rc2"
        ;;
    4.15.0)
        switch_to "4.15.0"
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        echo "Error: Unknown version '$1'"
        usage
        exit 1
        ;;
esac
