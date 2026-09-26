"""The DEAP fitness / individual classes shared by the GA trainers, the checkpoint
loader and the notebook.

DEAP registers these classes dynamically on `deap.creator`. Defining them in ONE
place guarantees that an individual restored from a checkpoint is exactly the
type the trainers produce (same base class, same fitness weights). Every function
is idempotent, so it is safe to call it from several places and from every worker
process.
"""
from deap import base, creator


def ensure_multi_objective_types() -> None:
    """MORSE (NSGA-II): `FitnessMulti` -- AUC and sign consistency, both
    maximised -- and `Individual`, a list of 0/1 feature-mask bits."""
    if "FitnessMulti" not in creator.__dict__:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))

    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMulti)


def ensure_single_objective_types() -> None:
    """SO-GA (AUC only): `FitnessSingle` -- AUC, maximised -- and
    `IndividualSingle`, a list of 0/1 feature-mask bits."""
    if "FitnessSingle" not in creator.__dict__:
        creator.create("FitnessSingle", base.Fitness, weights=(1.0,))

    if "IndividualSingle" not in creator.__dict__:
        creator.create("IndividualSingle", list, fitness=creator.FitnessSingle)
