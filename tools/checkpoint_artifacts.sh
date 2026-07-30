#!/usr/bin/env bash
#
# Commit and push the durable artifacts as training produces them.
#
# Tracking them in git (see .gitignore) means a container restart can no
# longer destroy them -- but only once they have actually been pushed.
# A restart that lands between round 7 and a manual commit still costs
# every round since the last push, and training rounds here run 40
# minutes to 3 hours, so "I'll commit it later" is a real gamble. This
# closes that window without touching the training loop itself: it
# watches for the signal that a round finished and pushes on its own.
#
# The trigger is models/*_parallel_state.json changing content, because
# train_navigation_parallel.train() writes it via save_progress() as the
# *last* thing it does in a round -- after merging the worker tables into
# the Q-table and after the demo episode. Seeing a new state file
# therefore means the Q-table beside it is already complete, which
# watching the Q-table directly would not guarantee.
#
# Nothing here is allowed to disturb training. Every git call is
# best-effort: a failed push logs and retries on the next tick rather
# than exiting, and JSON artifacts are parsed before they're staged so a
# commit can never capture a half-written file (json.dump is not atomic,
# so a tick landing mid-write would otherwise push invalid JSON and make
# the checkpoint useless for resuming).

set -uo pipefail

cd "$(dirname "$0")/.."

INTERVAL="${CHECKPOINT_INTERVAL:-60}"
BRANCH="$(git branch --show-current)"

# The paths .gitignore now tracks. Listed explicitly rather than using
# `git add -A` so this can never sweep up an unrelated edit someone is
# in the middle of making to source.
ARTIFACT_PATHS=(
    "models"
    "saves"
    "screenshots"
)

log() {
    echo "[$(date -u +%H:%M:%S)] $*"
}

# True only if every tracked JSON artifact currently parses. Guards
# against staging a file mid-write.
json_artifacts_are_intact() {
    local file
    while IFS= read -r file; do
        [ -f "$file" ] || continue
        python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$file" 2>/dev/null || return 1
    done < <(git ls-files --others --modified --exclude-standard -- \
             'models/*.json' 'screenshots/*_meta.json' 2>/dev/null)
    return 0
}

# A content fingerprint of the resume-critical progress files, so a
# round that merely rewrote identical numbers doesn't produce an empty
# commit.
progress_fingerprint() {
    cat models/*_parallel_state.json screenshots/*_meta.json 2>/dev/null | sha256sum | cut -d' ' -f1
}

# What round each training run is on, for the commit message -- the
# whole point of the checkpoint is which round it restores to.
progress_summary() {
    python3 - <<'PY' 2>/dev/null || echo "progress"
import glob, json, os
parts = []
for path in sorted(glob.glob("models/*_parallel_state.json")):
    name = os.path.basename(path).replace("_parallel_state.json", "")
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        continue
    parts.append(
        f"{name} round {data['round']} "
        f"({data['successes']}/{data['total_episodes']} successes, "
        f"epsilon {data['epsilon']:.3f})"
    )
print("; ".join(parts) if parts else "progress")
PY
}

push_with_retries() {
    local delay=2
    for _ in 1 2 3 4; do
        if git push -q -u origin "$BRANCH" 2>/dev/null; then
            return 0
        fi
        # A rejected push usually means someone else (a collaborating
        # session) pushed code to this branch. Without this, every push
        # after theirs fails non-fast-forward forever, silently -- which is
        # exactly how a full night of rounds once piled up locally while
        # origin sat frozen and the remote monitor concluded the machine
        # was asleep. Rebase our checkpoint commits (models/ artifacts,
        # which nobody else writes) onto their code commits and retry;
        # if the rebase itself fails, abort it cleanly and let the next
        # tick try again rather than wedging the worktree.
        git fetch -q origin "$BRANCH" 2>/dev/null || true
        if ! git rebase -q "origin/$BRANCH" 2>/dev/null; then
            git rebase --abort 2>/dev/null || true
            log "push rejected and rebase failed; will retry next tick"
        fi
        sleep "$delay"
        delay=$((delay * 2))
    done
    return 1
}

checkpoint_once() {
    if ! json_artifacts_are_intact; then
        log "skipping: a JSON artifact is mid-write"
        return
    fi

    git add -- "${ARTIFACT_PATHS[@]}" 2>/dev/null

    if git diff --cached --quiet 2>/dev/null; then
        return
    fi

    local summary
    summary="$(progress_summary)"
    local count
    count="$(git diff --cached --name-only | wc -l | tr -d ' ')"

    if git commit -q -m "Checkpoint training artifacts: ${summary}" \
        -m "Pushed automatically by tools/checkpoint_artifacts.sh so a container restart costs at most one round. ${count} file(s)."; then
        if push_with_retries; then
            log "pushed: ${summary}"
        else
            log "committed but push failed; will retry next tick"
        fi
    else
        log "commit failed; leaving files staged for next tick"
    fi
}

log "watching for round completions on ${BRANCH} (every ${INTERVAL}s)"
last_fingerprint="$(progress_fingerprint)"

# Push whatever is already uncommitted at startup, so arming this after
# a round has already finished doesn't wait for the next one.
checkpoint_once

while true; do
    sleep "$INTERVAL"
    current_fingerprint="$(progress_fingerprint)"
    if [ "$current_fingerprint" != "$last_fingerprint" ]; then
        last_fingerprint="$current_fingerprint"
        checkpoint_once
    fi
done
