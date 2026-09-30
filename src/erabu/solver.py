"""The SciPy/HiGHS boundary; no domain-specific modeling lives here."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp

from .compiler import CompiledModel


@dataclass(frozen=True)
class SolverOutput:
    status: int
    message: str
    x: NDArray[np.float64] | None
    objective: float | None
    bound: float | None
    gap: float | None


def run_solver(
    model: CompiledModel, *, time_limit: float | None, node_limit: int | None
) -> SolverOutput:
    options: dict[str, float | int] = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if node_limit is not None:
        options["node_limit"] = node_limit
    constraints = (
        LinearConstraint(model.matrix, model.lower, model.upper) if len(model.lower) else None
    )
    result = milp(
        model.objective,
        integrality=model.integrality,
        bounds=Bounds(0, 1),
        constraints=constraints,
        options=options,
    )
    return SolverOutput(
        int(result.status),
        str(result.message),
        result.x,
        result.fun,
        result.get("mip_dual_bound"),
        result.get("mip_gap"),
    )
