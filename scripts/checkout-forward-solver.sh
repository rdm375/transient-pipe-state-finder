#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$ROOT/cmake/ForwardSolverRevision.cmake"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Missing dependency manifest" >&2
    exit 1
fi

# Extract quoted values from the dependency manifest.
# The manifest uses one quoted value per CMake set() command.

REPOSITORY="$(
    sed -n '/STATE_FINDER_FORWARD_SOLVER_REPOSITORY/{n;s/^[[:space:]]*"\([^"]*\)".*/\1/p;}' "$MANIFEST"
)"

REVISION="$(
    sed -n '/STATE_FINDER_FORWARD_SOLVER_REVISION/{n;s/^[[:space:]]*"\([^"]*\)".*/\1/p;}' "$MANIFEST"
)"

if [[ -z "$REPOSITORY" ]]; then
    echo "ERROR: Missing repository URL" >&2
    exit 1
fi

if [[ ! "$REVISION" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: Invalid revision: $REVISION" >&2
    exit 1
fi

DESTINATION="${1:-$ROOT/external/transient-pipe-flow}"

if [[ -e "$DESTINATION" ]]; then
    echo "ERROR: Destination exists: $DESTINATION" >&2
    exit 1
fi

mkdir -p "$(dirname "$DESTINATION")"

echo "Cloning forward solver..."
echo "Repository: $REPOSITORY"
echo "Revision:   $REVISION"
echo "Destination: $DESTINATION"

git clone "$REPOSITORY" "$DESTINATION"
git -C "$DESTINATION" checkout --detach "$REVISION"

ACTUAL="$(git -C "$DESTINATION" rev-parse HEAD)"

if [[ "$ACTUAL" != "$REVISION" ]]; then
    echo "ERROR: Revision mismatch" >&2
    exit 1
fi

if [[ -n "$(git -C "$DESTINATION" status --porcelain)" ]]; then
    echo "ERROR: Checkout is not clean" >&2
    exit 1
fi

echo
echo "PASS: Forward solver checked out successfully."
echo "Commit: $ACTUAL"
