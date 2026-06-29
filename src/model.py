from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .scenarios import Scenario, apply_artificial_barrier


@dataclass
class SimulationResult:
    name: str
    description: str
    depth: np.ndarray
    times: np.ndarray
    steps: np.ndarray
    zeta: np.ndarray
    U: np.ndarray
    V: np.ndarray
    winds: np.ndarray
    dt: float
    config: ModelConfig

    @property
    def wet_mask(self) -> np.ndarray:
        return self.depth > 0.0


def gradient_x(field: np.ndarray, dx: float, wet_mask: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(field)
    if field.shape[0] < 2:
        return grad

    # Dry cells are barriers, not water points with zero surface elevation.
    field = np.where(wet_mask, field, 0.0)

    center_wet = wet_mask[1:-1, :]
    left_wet = wet_mask[:-2, :]
    right_wet = wet_mask[2:, :]
    interior = grad[1:-1, :]

    central = center_wet & left_wet & right_wet
    forward = center_wet & ~left_wet & right_wet
    backward = center_wet & left_wet & ~right_wet

    # Use centered differences in open water and one-sided differences next to land.
    interior[central] = (field[2:, :][central] - field[:-2, :][central]) / (2.0 * dx)
    interior[forward] = (field[2:, :][forward] - field[1:-1, :][forward]) / dx
    interior[backward] = (field[1:-1, :][backward] - field[:-2, :][backward]) / dx

    first = wet_mask[0, :] & wet_mask[1, :]
    last = wet_mask[-1, :] & wet_mask[-2, :]
    grad[0, first] = (field[1, first] - field[0, first]) / dx
    grad[-1, last] = (field[-1, last] - field[-2, last]) / dx
    grad[~wet_mask] = 0.0
    return grad


def gradient_y(field: np.ndarray, dy: float, wet_mask: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(field)
    if field.shape[1] < 2:
        return grad

    # Dry cells are barriers, not water points with zero surface elevation.
    field = np.where(wet_mask, field, 0.0)

    center_wet = wet_mask[:, 1:-1]
    lower_wet = wet_mask[:, :-2]
    upper_wet = wet_mask[:, 2:]
    interior = grad[:, 1:-1]

    central = center_wet & lower_wet & upper_wet
    forward = center_wet & ~lower_wet & upper_wet
    backward = center_wet & lower_wet & ~upper_wet

    # Use centered differences in open water and one-sided differences next to land.
    interior[central] = (field[:, 2:][central] - field[:, :-2][central]) / (2.0 * dy)
    interior[forward] = (field[:, 2:][forward] - field[:, 1:-1][forward]) / dy
    interior[backward] = (field[:, 1:-1][backward] - field[:, :-2][backward]) / dy

    first = wet_mask[:, 0] & wet_mask[:, 1]
    last = wet_mask[:, -1] & wet_mask[:, -2]
    grad[first, 0] = (field[first, 1] - field[first, 0]) / dy
    grad[last, -1] = (field[last, -1] - field[last, -2]) / dy
    grad[~wet_mask] = 0.0
    return grad


def _mask_dry_cells(field: np.ndarray, wet_mask: np.ndarray) -> np.ndarray:
    return np.where(wet_mask, field, 0.0)


def _staggered_face_masks(wet_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = wet_mask.shape
    u_mask = np.zeros((nx + 1, ny), dtype=bool)
    v_mask = np.zeros((nx, ny + 1), dtype=bool)
    u_mask[1:nx, :] = wet_mask[:-1, :] & wet_mask[1:, :]
    v_mask[:, 1:ny] = wet_mask[:, :-1] & wet_mask[:, 1:]
    return u_mask, v_mask


def _staggered_face_depths(depth: np.ndarray, u_mask: np.ndarray, v_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = depth.shape
    u_depth = np.ones((nx + 1, ny), dtype=float)
    v_depth = np.ones((nx, ny + 1), dtype=float)
    u_depth[1:nx, :] = 0.5 * (depth[:-1, :] + depth[1:, :])
    v_depth[:, 1:ny] = 0.5 * (depth[:, :-1] + depth[:, 1:])
    return np.where(u_mask, u_depth, 1.0), np.where(v_mask, v_depth, 1.0)


def _v_at_u_faces(V: np.ndarray, v_mask: np.ndarray, u_shape: tuple[int, int]) -> np.ndarray:
    sampled = np.zeros(u_shape, dtype=float)
    weights = np.zeros(u_shape, dtype=float)
    value_blocks = (V[:-1, :-1], V[:-1, 1:], V[1:, :-1], V[1:, 1:])
    mask_blocks = (v_mask[:-1, :-1], v_mask[:-1, 1:], v_mask[1:, :-1], v_mask[1:, 1:])
    for values, masks in zip(value_blocks, mask_blocks):
        sampled[1:-1, :] += np.where(masks, values, 0.0)
        weights[1:-1, :] += masks.astype(float)
    return np.divide(sampled, weights, out=np.zeros_like(sampled), where=weights > 0.0)


def _u_at_v_faces(U: np.ndarray, u_mask: np.ndarray, v_shape: tuple[int, int]) -> np.ndarray:
    sampled = np.zeros(v_shape, dtype=float)
    weights = np.zeros(v_shape, dtype=float)
    value_blocks = (U[:-1, :-1], U[1:, :-1], U[:-1, 1:], U[1:, 1:])
    mask_blocks = (u_mask[:-1, :-1], u_mask[1:, :-1], u_mask[:-1, 1:], u_mask[1:, 1:])
    for values, masks in zip(value_blocks, mask_blocks):
        sampled[:, 1:-1] += np.where(masks, values, 0.0)
        weights[:, 1:-1] += masks.astype(float)
    return np.divide(sampled, weights, out=np.zeros_like(sampled), where=weights > 0.0)


def transport_divergence(U: np.ndarray, V: np.ndarray, dx: float, dy: float, wet_mask: np.ndarray) -> np.ndarray:
    """Compute flux-form transport divergence with closed wet-land faces."""

    nx, ny = wet_mask.shape
    # Face arrays include the outer domain faces, which remain zero for closed boundaries.
    x_faces = np.zeros((nx + 1, ny), dtype=float)
    y_faces = np.zeros((nx, ny + 1), dtype=float)

    connected_x = wet_mask[:-1, :] & wet_mask[1:, :]
    connected_y = wet_mask[:, :-1] & wet_mask[:, 1:]
    # Only connected wet cells exchange transport; wet-land faces have zero normal flux.
    x_faces[1:nx, :] = np.where(connected_x, 0.5 * (U[:-1, :] + U[1:, :]), 0.0)
    y_faces[:, 1:ny] = np.where(connected_y, 0.5 * (V[:, :-1] + V[:, 1:]), 0.0)

    divergence = (x_faces[1:, :] - x_faces[:-1, :]) / dx + (y_faces[:, 1:] - y_faces[:, :-1]) / dy
    return np.where(wet_mask, divergence, 0.0)


def staggered_transport_divergence(U: np.ndarray, V: np.ndarray, dx: float, dy: float, wet_mask: np.ndarray) -> np.ndarray:
    """Compute cell-centered divergence from transports stored directly on cell faces."""

    divergence = (U[1:, :] - U[:-1, :]) / dx + (V[:, 1:] - V[:, :-1]) / dy
    return np.where(wet_mask, divergence, 0.0)


def initialize_state(depth: np.ndarray, config: ModelConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create zeta and transport arrays for the configured grid arrangement."""

    zeta = np.zeros(depth.shape, dtype=float)
    if config.grid_mode == "collocated":
        return zeta, np.zeros(depth.shape, dtype=float), np.zeros(depth.shape, dtype=float)
    if config.grid_mode == "staggered":
        nx, ny = depth.shape
        return zeta, np.zeros((nx + 1, ny), dtype=float), np.zeros((nx, ny + 1), dtype=float)
    raise ValueError(f"Unknown grid_mode: {config.grid_mode}")


def _step_forward_collocated(
    zeta: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    depth: np.ndarray,
    wind_x: float,
    wind_y: float,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance cell-centered zeta and transports by one explicit Euler time step."""

    wet_mask = depth > 0.0
    safe_depth = np.where(wet_mask, depth, 1.0)

    zeta_x = gradient_x(zeta, config.dx, wet_mask)
    zeta_y = gradient_y(zeta, config.dy, wet_mask)

    transport_speed = np.sqrt(U * U + V * V)
    quadratic_drag = config.friction * transport_speed / (safe_depth * safe_depth)
    wind_speed = float(np.hypot(wind_x, wind_y))

    dUdt = (
        -config.g * depth * zeta_x
        - quadratic_drag * U
        + config.wind_stress * wind_x * wind_speed
        + config.coriolis_f * V
    )
    dVdt = (
        -config.g * depth * zeta_y
        - quadratic_drag * V
        + config.wind_stress * wind_y * wind_speed
        - config.coriolis_f * U
    )

    next_U = _mask_dry_cells(U + config.dt * dUdt, wet_mask)
    next_V = _mask_dry_cells(V + config.dt * dVdt, wet_mask)

    div_transport = transport_divergence(next_U, next_V, config.dx, config.dy, wet_mask)
    next_zeta = np.where(wet_mask, zeta - config.dt * div_transport, 0.0)

    if not np.all(np.isfinite(next_zeta)) or not np.all(np.isfinite(next_U)) or not np.all(np.isfinite(next_V)):
        raise FloatingPointError("Simulation produced NaN or infinite values.")

    return next_zeta, next_U, next_V


def _step_forward_staggered(
    zeta: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    depth: np.ndarray,
    wind_x: float,
    wind_y: float,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance cell-centered zeta and face-centered transports by one explicit Euler time step."""

    wet_mask = depth > 0.0
    u_mask, v_mask = _staggered_face_masks(wet_mask)
    u_depth, v_depth = _staggered_face_depths(depth, u_mask, v_mask)

    zeta_x = np.zeros_like(U)
    zeta_y = np.zeros_like(V)
    zeta_x[1:-1, :] = np.where(u_mask[1:-1, :], (zeta[1:, :] - zeta[:-1, :]) / config.dx, 0.0)
    zeta_y[:, 1:-1] = np.where(v_mask[:, 1:-1], (zeta[:, 1:] - zeta[:, :-1]) / config.dy, 0.0)

    V_on_U = _v_at_u_faces(V, v_mask, U.shape)
    U_on_V = _u_at_v_faces(U, u_mask, V.shape)
    wind_speed = float(np.hypot(wind_x, wind_y))

    u_transport_speed = np.sqrt(U * U + V_on_U * V_on_U)
    v_transport_speed = np.sqrt(U_on_V * U_on_V + V * V)
    u_drag = config.friction * u_transport_speed / (u_depth * u_depth)
    v_drag = config.friction * v_transport_speed / (v_depth * v_depth)

    dUdt = (
        -config.g * u_depth * zeta_x
        - u_drag * U
        + config.wind_stress * wind_x * wind_speed
        + config.coriolis_f * V_on_U
    )
    dVdt = (
        -config.g * v_depth * zeta_y
        - v_drag * V
        + config.wind_stress * wind_y * wind_speed
        - config.coriolis_f * U_on_V
    )

    next_U = np.where(u_mask, U + config.dt * dUdt, 0.0)
    next_V = np.where(v_mask, V + config.dt * dVdt, 0.0)

    div_transport = staggered_transport_divergence(next_U, next_V, config.dx, config.dy, wet_mask)
    next_zeta = np.where(wet_mask, zeta - config.dt * div_transport, 0.0)

    if not np.all(np.isfinite(next_zeta)) or not np.all(np.isfinite(next_U)) or not np.all(np.isfinite(next_V)):
        raise FloatingPointError("Simulation produced NaN or infinite values.")

    return next_zeta, next_U, next_V


def step_forward(
    zeta: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    depth: np.ndarray,
    wind_x: float,
    wind_y: float,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance zeta and transports using the configured grid arrangement."""

    if config.grid_mode == "collocated":
        return _step_forward_collocated(zeta, U, V, depth, wind_x, wind_y, config)
    if config.grid_mode == "staggered":
        return _step_forward_staggered(zeta, U, V, depth, wind_x, wind_y, config)
    raise ValueError(f"Unknown grid_mode: {config.grid_mode}")


def run_simulation(
    scenario: Scenario,
    depth: np.ndarray,
    config: ModelConfig,
) -> SimulationResult:
    scenario_depth = apply_artificial_barrier(depth) if scenario.add_barrier else depth.copy()
    wet_mask = scenario_depth > 0.0
    zeta, U, V = initialize_state(scenario_depth, config)

    saved_zeta: list[np.ndarray] = []
    saved_U: list[np.ndarray] = []
    saved_V: list[np.ndarray] = []
    saved_times: list[float] = []
    saved_steps: list[int] = []
    saved_winds: list[tuple[float, float]] = []

    for step in range(config.steps + 1):
        if step % config.output_every == 0 or step == config.steps:
            saved_zeta.append(zeta.copy())
            saved_U.append(U.copy())
            saved_V.append(V.copy())
            saved_times.append(step * config.dt)
            saved_steps.append(step)
            saved_winds.append(scenario.wind(min(step, config.steps - 1)))

        if step == config.steps:
            break

        wind_x, wind_y = scenario.wind(step)
        zeta, U, V = step_forward(zeta, U, V, scenario_depth, wind_x, wind_y, config)
        zeta = np.where(wet_mask, zeta, 0.0)
        if config.grid_mode == "collocated":
            U = _mask_dry_cells(U, wet_mask)
            V = _mask_dry_cells(V, wet_mask)
        else:
            u_mask, v_mask = _staggered_face_masks(wet_mask)
            U = np.where(u_mask, U, 0.0)
            V = np.where(v_mask, V, 0.0)

    return SimulationResult(
        name=scenario.name,
        description=scenario.description,
        depth=scenario_depth,
        times=np.asarray(saved_times),
        steps=np.asarray(saved_steps),
        zeta=np.asarray(saved_zeta),
        U=np.asarray(saved_U),
        V=np.asarray(saved_V),
        winds=np.asarray(saved_winds),
        dt=config.dt,
        config=config,
    )
