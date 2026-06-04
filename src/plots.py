from __future__ import annotations

import os
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from .diagnostics import (
    eddy_kinetic_energy,
    masked_stats,
    net_transport_through_channel,
    velocity_components,
    vorticity,
)
from .model import SimulationResult


def _extent(result: SimulationResult) -> tuple[float, float, float, float]:
    nx, ny = result.depth.shape
    return (0.0, ny * result.config.dy / 1000.0, 0.0, nx * result.config.dx / 1000.0)


def _imshow(ax: plt.Axes, field: np.ndarray, result: SimulationResult, title: str, cmap: str = "viridis") -> None:
    im = ax.imshow(
        field,
        origin="lower",
        extent=_extent(result),
        aspect="auto",
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("y distance (km)")
    ax.set_ylabel("x distance (km)")
    plt.colorbar(im, ax=ax, shrink=0.82)


def _lake_axes(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = result.depth.shape
    y_km = np.arange(ny) * result.config.dy / 1000.0
    x_km = np.arange(nx) * result.config.dx / 1000.0
    return y_km, x_km


def _plot_stream_snapshot(ax: plt.Axes, result: SimulationResult, frame: int, title: str) -> None:
    y_km, x_km = _lake_axes(result)
    u, v = velocity_components(result)
    speed = np.sqrt(u[frame] ** 2 + v[frame] ** 2)
    wet = result.wet_mask
    u_frame = np.nan_to_num(v[frame], nan=0.0)
    v_frame = np.nan_to_num(u[frame], nan=0.0)
    color_field = np.nan_to_num(speed, nan=0.0)

    ax.contourf(y_km, x_km, wet.astype(float), levels=[0.5, 1.5], colors=["#f7fbff"], alpha=1.0)
    ax.contour(y_km, x_km, wet.astype(float), levels=[0.5], colors="#888888", linewidths=0.8)
    stream = ax.streamplot(
        y_km,
        x_km,
        u_frame,
        v_frame,
        color=color_field,
        cmap="Spectral_r",
        density=1.35,
        linewidth=1.2,
        arrowsize=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("y distance (km)")
    ax.set_ylabel("x distance (km)")
    ax.set_xlim(y_km.min(), y_km.max())
    ax.set_ylim(x_km.min(), x_km.max())
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(stream.lines, ax=ax, shrink=0.82, label="velocity (m/s)")


def plot_point_timeseries(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    for result in results:
        i, j = result.config.point_index
        ax.plot(result.times / 3600.0, result.zeta[:, i, j], label=result.name)
    ax.set_title("Sea level zeta at grid coordinate [25, 10]")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("zeta (m)")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_mean_std_maps(result: SimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    u, v = velocity_components(result)
    fields = {
        "zeta": result.zeta,
        "u": u,
        "v": v,
    }
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), dpi=160)
    for row, (name, field) in enumerate(fields.items()):
        mean_field, std_field = masked_stats(field, result.wet_mask)
        _imshow(axes[row, 0], mean_field, result, f"{name} mean")
        _imshow(axes[row, 1], std_field, result, f"{name} standard deviation", cmap="magma")
    fig.suptitle(f"{result.name}: mean and standard deviation maps")
    fig.tight_layout()
    fig.savefig(output_dir / f"{result.name}_mean_std_maps.png")
    plt.close(fig)


def plot_hovmoller(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, result in zip(axes, results):
        x_index = result.config.transect_x_index
        transect = result.zeta[:, x_index, :]
        im = ax.imshow(
            transect,
            origin="lower",
            aspect="auto",
            extent=(0, result.depth.shape[1] - 1, result.times[0] / 3600.0, result.times[-1] / 3600.0),
            cmap="coolwarm",
        )
        ax.set_title(result.name)
        ax.set_xlabel("y index")
        ax.set_ylabel("Time (hours)")
        plt.colorbar(im, ax=ax, shrink=0.82, label="zeta (m)")
    fig.suptitle("Hovmoller plots for zeta along x transect 25")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_flow_snapshots(result: SimulationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = sorted({0, len(result.times) // 3, 2 * len(result.times) // 3, len(result.times) - 1})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=160)
    axes = axes.ravel()
    yy, xx = np.meshgrid(np.arange(result.depth.shape[1]), np.arange(result.depth.shape[0]))
    u, v = velocity_components(result)
    for ax, idx in zip(axes, frame_indices):
        zeta_frame = np.where(result.wet_mask, result.zeta[idx], np.nan)
        im = ax.imshow(zeta_frame, origin="lower", aspect="auto", cmap="coolwarm")
        stride = 2
        ax.quiver(
            yy[::stride, ::stride],
            xx[::stride, ::stride],
            u[idx, ::stride, ::stride],
            v[idx, ::stride, ::stride],
            color="black",
            scale=0.04,
            width=0.003,
        )
        ax.set_title(f"t={result.times[idx] / 3600.0:.2f} h")
        ax.set_xlabel("y index")
        ax.set_ylabel("x index")
        plt.colorbar(im, ax=ax, shrink=0.75, label="zeta (m)")
    fig.suptitle(f"{result.name}: flow snapshots")
    fig.tight_layout()
    fig.savefig(output_dir / f"{result.name}_flow_snapshots.png")
    plt.close(fig)


def plot_presentation_style(result: SimulationResult, output_dir: Path) -> None:
    """Create a cleaner slide-style figure inspired by the lecture screenshot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    early = max(1, len(result.times) // 4)
    late = len(result.times) - 1
    transport = net_transport_through_channel(result)

    fig = plt.figure(figsize=(16, 8.5), dpi=160)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.12, 0.88],
        width_ratios=[1.0, 1.0, 1.45],
        left=0.045,
        right=0.98,
        top=0.94,
        bottom=0.08,
        wspace=0.26,
    )
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.55,
        "The Semester Project: Model of a Lake",
        fontsize=24,
        fontweight="bold",
        va="center",
        ha="left",
    )
    title_ax.text(1.0, 0.58, "TUM", fontsize=20, fontweight="bold", color="#0065bd", va="center", ha="right")

    ax1 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[1, 1])
    ax3 = fig.add_subplot(grid[1, 2])
    _plot_stream_snapshot(ax1, result, early, f"{result.name}: t={result.times[early] / 3600.0:.2f} h")
    _plot_stream_snapshot(ax2, result, late, f"{result.name}: t={result.times[late] / 3600.0:.2f} h")

    ax3.plot(result.times / 3600.0, transport, color="#1f77b4", linewidth=2.0)
    ax3.set_title("Net Transport through Channel", fontsize=17)
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("transport (m3/s)")
    ax3.grid(True, color="white", linewidth=1.2)
    ax3.set_facecolor("#e9e9f2")

    fig.text(0.30, 0.035, "Velocity snapshots!", fontsize=17, ha="center")
    fig.text(
        0.045,
        0.025,
        "Deutsches Geodätisches Forschungsinstitut (DGFI-TUM) | Technische Universität München",
        fontsize=11,
        color="#0065bd",
        ha="left",
    )
    fig.savefig(output_dir / f"{result.name}_presentation_style.png")
    plt.close(fig)


def animate_lake(result: SimulationResult, output_dir: Path, max_frames: int = 90) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = np.linspace(0, len(result.times) - 1, min(max_frames, len(result.times)), dtype=int)
    y_km, x_km = _lake_axes(result)
    u, v = velocity_components(result)
    zeta = np.where(result.wet_mask, result.zeta, np.nan)
    zmax = float(np.nanmax(np.abs(zeta)))
    if zmax == 0.0:
        zmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 9), dpi=120)
    im = ax.imshow(
        zeta[0],
        origin="lower",
        extent=_extent(result),
        aspect="equal",
        cmap="coolwarm",
        vmin=-zmax,
        vmax=zmax,
    )
    ax.contour(y_km, x_km, result.wet_mask.astype(float), levels=[0.5], colors="#333333", linewidths=0.8)
    stride = 2
    yy, xx = np.meshgrid(y_km, x_km)
    quiver = ax.quiver(
        yy[::stride, ::stride],
        xx[::stride, ::stride],
        np.nan_to_num(v[0, ::stride, ::stride]),
        np.nan_to_num(u[0, ::stride, ::stride]),
        color="black",
        scale=0.35,
        width=0.003,
    )
    title = ax.set_title("")
    ax.set_xlabel("y distance (km)")
    ax.set_ylabel("x distance (km)")
    plt.colorbar(im, ax=ax, shrink=0.78, label="zeta (m)")

    def update(frame_id: int):
        im.set_data(zeta[frame_id])
        quiver.set_UVC(
            np.nan_to_num(v[frame_id, ::stride, ::stride]),
            np.nan_to_num(u[frame_id, ::stride, ::stride]),
        )
        wx, wy = result.winds[frame_id]
        title.set_text(f"{result.name} | t={result.times[frame_id] / 3600.0:.2f} h | wind=({wx:.0f}, {wy:.0f}) m/s")
        return im, quiver, title

    animation = matplotlib.animation.FuncAnimation(fig, update, frames=frame_ids, interval=90, blit=False)
    animation.save(output_dir / f"{result.name}_lake_animation.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


def animate_scenario_comparison(results: list[SimulationResult], output_dir: Path, max_frames: int = 90) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = min(max_frames, *(len(result.times) for result in results))
    frame_ids = np.linspace(0, min(len(result.times) for result in results) - 1, frame_count, dtype=int)
    zmax = max(float(np.nanmax(np.abs(np.where(result.wet_mask, result.zeta, np.nan)))) for result in results)
    if zmax == 0.0:
        zmax = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(8, 10), dpi=120)
    axes = axes.ravel()
    images = []
    for ax, result in zip(axes, results):
        im = ax.imshow(
            np.where(result.wet_mask, result.zeta[0], np.nan),
            origin="lower",
            extent=_extent(result),
            aspect="equal",
            cmap="coolwarm",
            vmin=-zmax,
            vmax=zmax,
        )
        y_km, x_km = _lake_axes(result)
        ax.contour(y_km, x_km, result.wet_mask.astype(float), levels=[0.5], colors="#333333", linewidths=0.7)
        ax.set_title(result.name)
        ax.set_xlabel("y distance (km)")
        ax.set_ylabel("x distance (km)")
        images.append(im)
    fig.colorbar(images[0], ax=axes.tolist(), shrink=0.72, label="zeta (m)")
    fig.suptitle("Lake sea-level response through time")

    def update(frame_id: int):
        for im, result in zip(images, results):
            im.set_data(np.where(result.wet_mask, result.zeta[frame_id], np.nan))
        fig.suptitle(f"Lake sea-level response through time | t={results[0].times[frame_id] / 3600.0:.2f} h")
        return images

    animation = matplotlib.animation.FuncAnimation(fig, update, frames=frame_ids, interval=90, blit=False)
    animation.save(output_dir / "all_scenarios_lake_animation.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


def plot_vorticity_eke(results: list[SimulationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [result.name for result in results]
    mean_vort = [float(np.nanmean(vorticity(result))) for result in results]
    mean_eke = [float(np.nanmean(eddy_kinetic_energy(result))) for result in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
    axes[0].bar(names, mean_vort, color="#3B7EA1")
    axes[0].set_title("Mean vorticity")
    axes[0].set_ylabel("s^-1")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(names, mean_eke, color="#B85C38")
    axes[1].set_title("Mean eddy kinetic energy")
    axes[1].set_ylabel("m^2 s^-2")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all(results: list[SimulationResult], output_dir: Path, make_animations: bool = False) -> None:
    figures_dir = output_dir / "figures"
    animations_dir = output_dir / "animations"
    plot_point_timeseries(results, figures_dir / "question_a_point_timeseries.png")
    plot_hovmoller(results, figures_dir / "question_c_hovmoller_transect_25.png")
    plot_vorticity_eke(results, figures_dir / "question_e_vorticity_eke.png")
    for result in results:
        plot_mean_std_maps(result, figures_dir)
        plot_flow_snapshots(result, figures_dir)
        plot_presentation_style(result, figures_dir)
        if make_animations:
            animate_lake(result, animations_dir)
    if make_animations:
        animate_scenario_comparison(results, animations_dir)
