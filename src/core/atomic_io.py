import json
import os
import tempfile


def write_json_atomic(path, data, **dump_kwargs):
    """
    Write JSON so a reader never sees a partial file.

    `json.dump` into an opened file is not atomic: it truncates the target
    immediately and then streams into it, so anything reading during the
    write sees a valid-looking file that is simply cut off partway. For
    most of this project that window is too small to matter -- but two
    things here read these files while something else is writing them, and
    both hit it for real:

      - A container restart mid-survey left screenshots/
        forest_map_meta.json truncated. rewards/forest_rewards.py reads it
        at import time, so the damage surfaced as the forest environment
        refusing to import at all rather than as a bad number somewhere.
      - tools/checkpoint_artifacts.sh commits the Q-table and progress
        files while training keeps running, so a tick landing mid-write
        would push a truncated checkpoint -- one that looks committed and
        safe but cannot actually be resumed from.

    Writing to a temporary file in the same directory and then renaming is
    the standard fix: os.replace is atomic on POSIX and Windows alike, so
    the destination only ever contains the previous complete version or
    the new complete one. Same directory specifically, because rename is
    only atomic within a filesystem.

    fsync before the rename makes it durable across a crash rather than
    only across a concurrent read. Without it the rename can reach disk
    while the data behind it is still buffered, which is precisely the
    "file exists but is truncated" state this is meant to prevent.
    """
    path = str(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(data, handle, **dump_kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        # Leaving a stray .tmp behind would be harmless, but it would also
        # be indistinguishable from a write still in progress.
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
