"""
HINTS hierarchies with a different branch factor at each level, in the manner
of Chapter 2's hierarchy study (its [1,2,4] beside [1,8], [2,4], [4,2], ...),
on SUSY and HEPMASS at the budget of experiments_budget.py.

Hierarchies are written as in Chapter 2, [m_0, m_1, ..., m_H]: m_0 blocks at
each unit vertex and branch factor m_h at level h, counted up from the unit
vertices. Every hierarchy here is over 64 blocks of 1/64 of the rows below the
subtree root (ss_prop = 1/64), the split of experiments_budget_hierarchies.py.
On trees a unit vertex of m_0 blocks is the same as one block of m_0/64 of the
rows, since the blocks are an equal split in random order; the hierarchies with
one branch factor throughout are in experiments_budget_hierarchies.py.

The six were chosen before the runs, to vary the size of the unit vertices (1,
2, 4 and 8 blocks) and the order of the branch factors:

    [1,2,4,8]   single-block units, widest split at the top
    [1,8,4,2]   single-block units, widest split at the bottom
    [2,4,8]     two-block units, widening upwards
    [4,2,8]     four-block units
    [8,2,4]     eight-block units, as Chapter 2's [1,2,4]
    [8,4,2]     eight-block units, the other order

Every hierarchy has the same ss_prop, so its file names would clash with the
others', and each is a run of its own, read as GROUPS[name] through
`--file <this file>::<name>`:

    E=examples/incremental_decision_tree/experiments_budget_mixed_hierarchies.py
    for h in 1-2-4-8 1-8-4-2 2-4-8 4-2-8 8-2-4 8-4-2; do
        PREP_CORES=4 CORES=64 TIME=8:00:00 EVAL_CORES=64 EVAL_TIME=24:00:00 \\
        RUN_ID=ch3_hier_$h EXPERIMENTS="$E::$h" slurm/submit.sh
    done
"""
from experiments_budget_hierarchies import DATASETS, entry

HIERARCHIES = ([1, 2, 4, 8], [1, 8, 4, 2], [2, 4, 8], [4, 2, 8], [8, 2, 4], [8, 4, 2])

GROUPS = {"-".join(map(str, h)): [entry(d, "HINTS", 1 / 64, hierarchy=list(h))
                                  for d in DATASETS]
          for h in HIERARCHIES}

# The first, so that the file also works without ::name.
EXPERIMENTS = GROUPS["1-2-4-8"]
