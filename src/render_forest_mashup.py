"""
Generate forest rollouts and render the mashup GIF in one step.

    cd src && ../.venv/bin/python3 render_forest_mashup.py [num_runs]

The deliverable this whole line of work has been building toward: every
rollout of the trained forest agent drawn walking the surveyed panorama
at once -- green for runs that reached the Pewter-side exit, red for
ones that didn't, yellow while still moving.

Thin by design: rollouts come from generate_forest_mashup_rollouts, and
the drawing is render_route1_mashup.main, which was already
parameterized by map prefix and rollout filename -- the forest only had
to supply its own env and assets, not a second renderer.

Runnable at any training stage. Before convergence it renders an
honestly-red swarm (useful as a progress snapshot); once the Q-table
solves the maze, the same command produces the showcase GIF.
"""

import sys

import generate_forest_mashup_rollouts
import render_route1_mashup


def main(num_runs=150):
    run_label = generate_forest_mashup_rollouts.main(num_runs=num_runs)
    render_route1_mashup.main(
        run_label=run_label,
        map_prefix="forest",
        rollouts_name="forest_mashup_rollouts.json",
        gif_name="forest_mashup.gif",
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
