"""
Check that the merge weights worker contributions by how often each value
was actually updated, instead of diluting rare learning with inherited
copies.

    .venv/bin/python3 tools/test_weighted_merge.py

The scenario is the one observed in the 18-worker forest run: every worker
starts a round holding the same shared table, one worker explores a deep
tile and learns a strongly different value for it, and the other N-1 never
touch it. A plain average moves the merged value 1/N of the way toward
what was learned -- with 18 workers, five consecutive greedy demos froze
at exactly 115 tiles while training successes kept climbing, because the
deep-maze frontier was being averaged back toward its old values on every
merge.

Cases:

  - One worker updated, N-1 inherited: the merged value must be the
    updating worker's value exactly, not 1/N of the way there.
  - Two workers updated with different visit counts: merged value must be
    the count-weighted mean, so heavier evidence counts for more.
  - Nobody updated: the inherited value must carry through untouched.
  - A worker with no counts sidecar (older format): weight-1 fallback,
    which reproduces the old equal-weight behavior rather than crashing.
"""

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from train_navigation_parallel import merge_tables

failures = []


def check(label, ok, fail_detail="", ok_detail=""):
    detail = ok_detail if ok else fail_detail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def write_worker(directory, name, table, counts):
    path = directory / f"{name}.json"
    path.write_text(json.dumps(table))
    if counts is not None:
        pathlib.Path(str(path) + ".counts").write_text(json.dumps(counts))
    return path


def main():
    d = pathlib.Path(tempfile.mkdtemp())
    key = "51,10,10"
    inherited = [0.5, 0.5, 0.5, 0.5]

    # 18 workers: one learned action 2 at the deep tile (visited 40 times),
    # the rest carry the inherited row untouched.
    paths = []
    learned = [0.5, 0.5, 9.0, 0.5]
    paths.append(write_worker(d, "worker0", {key: learned}, {f"{key}|2": 40}))
    for i in range(1, 18):
        paths.append(write_worker(d, f"worker{i}", {key: inherited}, {}))

    out = d / "merged.json"
    merge_tables(paths, out)
    merged = json.loads(out.read_text())

    check(
        "one worker's learning is not diluted by 17 inherited copies",
        merged[key][2] == 9.0,
        fail_detail=f"got {merged[key][2]} -- plain averaging would give "
                    f"{(9.0 + 17 * 0.5) / 18:.3f}",
        ok_detail="merged value is exactly the learned 9.0",
    )
    check(
        "untouched actions carry the inherited value",
        merged[key][0] == 0.5,
        fail_detail=f"got {merged[key][0]}",
        ok_detail="0.5 preserved",
    )

    # Two workers with different visit counts on the same action.
    d2 = pathlib.Path(tempfile.mkdtemp())
    paths = [
        write_worker(d2, "worker0", {key: [0.5, 3.0, 0.5, 0.5]}, {f"{key}|1": 30}),
        write_worker(d2, "worker1", {key: [0.5, 6.0, 0.5, 0.5]}, {f"{key}|1": 10}),
    ]
    out2 = d2 / "merged.json"
    merge_tables(paths, out2)
    merged2 = json.loads(out2.read_text())
    expected = (3.0 * 30 + 6.0 * 10) / 40
    check(
        "two updaters merge by visit-weighted mean",
        abs(merged2[key][1] - expected) < 1e-9,
        fail_detail=f"got {merged2[key][1]}, expected {expected}",
        ok_detail=f"{merged2[key][1]:.3f} (=30:10 weighting), not the plain mean 4.5",
    )

    # Legacy worker without a counts file: falls back to weight 1.
    d3 = pathlib.Path(tempfile.mkdtemp())
    paths = [
        write_worker(d3, "worker0", {key: [1.0, 0, 0, 0]}, None),
        write_worker(d3, "worker1", {key: [3.0, 0, 0, 0]}, None),
    ]
    out3 = d3 / "merged.json"
    merge_tables(paths, out3)
    merged3 = json.loads(out3.read_text())
    check(
        "counts-less workers fall back to the old equal-weight behavior",
        merged3[key][0] == 2.0,
        fail_detail=f"got {merged3[key][0]}",
        ok_detail="plain mean 2.0, no crash",
    )

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        return 1
    print("Rare deep-maze learning now survives the merge at full strength")
    print("instead of arriving at 1/num_workers of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
