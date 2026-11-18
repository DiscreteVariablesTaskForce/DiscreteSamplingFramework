"""
The shape of the HINTS hierarchy, on susy at ss_prop 1/16: the trees' version
of Chapter 2's hierarchy study.

HINTS has one branching factor per level, so a hierarchy has branching**levels
unit blocks, and only shapes that factor the block count exactly keep the unit
blocks at ss_prop of the rows. At 1/16 there are three:

    4 levels x 2   deep and binary      experiments.py, the main comparison
    2 levels x 4   shallow and wide     here
    1 level  x 16  flat                 not run: FlatHINTS at 1/16, in
                                        experiments.py, is all but this shape

At 1/4, 1/8 and 1/32 the only alternative is the flat one. The file names
carry ss_prop but not the shape, so this goes in a run of its own, and one
shape per run:

    RUN_ID=hints_2x4 EXPERIMENTS=examples/incremental_decision_tree/experiments_hints_shapes.py \
        slurm/submit.sh
"""
from experiments import MODEL, runs

SS_PROP = 1 / 16
LEVELS, BRANCHING = 2, 4

EXPERIMENTS = [dict(dataset="susy", samplers=[sampler], proposals=["HINTS"],
                    ss_prop=SS_PROP, levels=LEVELS, branching=BRANCHING,
                    **MODEL, **run)
               for sampler, run in runs("susy", "HINTS").items()]
