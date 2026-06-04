from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


WindFunction = Callable[[int], tuple[float, float]]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    wind: WindFunction
    add_barrier: bool = False


def scenario_1_wind(step: int) -> tuple[float, float]:
    return 10.0, 0.0


def scenario_2_wind(step: int) -> tuple[float, float]:
    if step < 50:
        return 0.0, 10.0
    if step < 250:
        return 10.0, 0.0
    return 0.0, 0.0


def scenario_3_wind(step: int) -> tuple[float, float]:
    if step < 50:
        return 10.0, 5.0
    if step < 200:
        return 10.0, 5.0
    return 0.0, 0.0


SCENARIOS = (
    Scenario(
        name="scenario_1",
        description="Constant 10 m/s easterly wind for the full model run.",
        wind=scenario_1_wind,
    ),
    Scenario(
        name="scenario_2",
        description="10 m/s northward wind for 50 steps, 10 m/s easterly wind for 200 steps, then no wind.",
        wind=scenario_2_wind,
    ),
    Scenario(
        name="scenario_3",
        description="10 m/s easterly and 5 m/s northerly wind for 200 steps, then no wind.",
        wind=scenario_3_wind,
    ),
    Scenario(
        name="scenario_4",
        description="Scenario 3 with an artificial barrier along the middle y transect.",
        wind=scenario_3_wind,
        add_barrier=True,
    ),
)


def apply_artificial_barrier(depth: np.ndarray) -> np.ndarray:
    """Set the middle y transect to land, splitting the lake into two basins."""

    modified = depth.copy()
    middle_y = modified.shape[1] // 2
    modified[:, middle_y] = 0.0
    return modified

