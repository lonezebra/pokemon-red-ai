#!/bin/bash
#
# Back up the training artifacts git deliberately does not hold, to a
# network share mounted under /Volumes.
#
# The repo's .gitignore keeps the whole-game checkpoint history out of git
# on purpose (3,500 near-identical 7.8MB blobs would push tens of GB into
# history, permanently); the four milestone policies are committed, but the
# full history -- the thing that lets any experiment resume from any point,
# and the raw material every report's bisection ran over -- exists only on
# this machine until it is copied somewhere. That is what this script is
# for.
#
# rsync, not cp: the first run moves ~25GB, every later run moves only the
# checkpoints written since. Never --delete: a backup that can remove things
# from the destination is a mirror, and a mirror faithfully reproduces the
# accident you were keeping a backup against.
#
#   tools/backup_training_artifacts.sh /Volumes/<ShareName>
#   tools/backup_training_artifacts.sh /Volumes/<ShareName> --dry-run
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_ROOT="${1:-}"
shift || true

if [ -z "$DEST_ROOT" ]; then
    echo "usage: $0 /Volumes/<ShareName> [--dry-run]" >&2
    echo "Mount the share first (Finder -> Cmd+K), then pass its mount point." >&2
    exit 2
fi

# A share that dropped mid-session leaves /Volumes/<name> as a plain, empty
# local directory -- rsync into that "succeeds" by filling the boot disk
# while backing up nothing. Refuse anything that is not a real mount point.
if ! mount | grep -q " ${DEST_ROOT} "; then
    echo "error: ${DEST_ROOT} is not a mounted volume." >&2
    echo "Mount the share (Finder -> Cmd+K) and re-run." >&2
    exit 1
fi

DEST="${DEST_ROOT}/pokemon-red-ai-backup"
mkdir -p "$DEST"

# Everything the repo's .gitignore excludes on size grounds, in one place:
#   models/            checkpoint history, rollout JSONs, tensorboard, the
#                      pre-milestone snapshot dirs, plus the (small,
#                      also-committed) milestone zips riding along
#   screenshots/mashups/  per-checkpoint eval renders, incl. whole_game_*
# Deliberately NOT backed up: training_scratch (per-round demo frames the
# next round supersedes -- the same judgment .gitignore already made).
SOURCES=(
    "models"
    "screenshots/mashups"
)

echo "Backing up to ${DEST}"
echo "Sources: ${SOURCES[*]}"
echo

for src in "${SOURCES[@]}"; do
    echo "=== ${src} ==="
    rsync -a --partial --human-readable --stats \
        --exclude ".DS_Store" \
        --exclude "training_scratch/" \
        "$@" \
        "${PROJECT_ROOT}/${src}/" "${DEST}/${src}/" \
        | grep -E "Number of created files|Number of regular files transferred|Total transferred file size|Total file size" \
        || true
    echo
done

echo "=== destination totals ==="
du -sh "${DEST}"/* 2>/dev/null || true
echo
echo "Backup complete: $(date '+%Y-%m-%d %H:%M:%S')"
