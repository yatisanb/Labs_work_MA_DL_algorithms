from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_CONV_COLORS = [
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#e9c46a",
    "#8338ec",
    "#fb5607",
    "#3a86ff",
]
_TRAJ_COLORS = ["#ff4d6d", "#ffd166", "#06d6a0", "#118ab2", "#8338ec", "#fb5607"]


def _draw_convergence(
    ax: plt.Axes,
    func: Callable[[np.ndarray], np.ndarray],
    paths: list[np.ndarray],
    labels: list[str | None],
    yscale: Literal["log", "symlog", "linear"] | None,
    title: str,
) -> None:
    """Draw convergence curves onto an existing Axes."""
    all_values = []
    for path in paths:
        path = np.array(path)
        vals = np.array([func(pt) for pt in path]) if path.ndim == 2 else func(path)
        all_values.append(vals)

    if yscale is None:
        flat = np.concatenate(all_values)
        if np.all(flat > 0):
            yscale = "log"
        elif np.any(flat < 0):
            yscale = "symlog"
        else:
            yscale = "linear"

    ylabel_map: dict[str, str] = {
        "log": "f(x)  (log-шкала)",
        "symlog": "f(x)  (symlog-шкала)",
        "linear": "f(x)",
    }

    ax.set_facecolor("#fbfbfb")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    for i, (vals, label) in enumerate(zip(all_values, labels)):
        color = _CONV_COLORS[i % len(_CONV_COLORS)]
        ax.plot(
            np.arange(len(vals)),
            vals,
            marker="o",
            linestyle="-",
            color=color,
            label=label,
            linewidth=2.5,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
        )

    if yscale == "log":
        ax.set_yscale("log")
    elif yscale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-3)

    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Итерация", fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel_map[yscale], fontsize=12, labelpad=8)
    ax.grid(which="major", linestyle="-", linewidth=0.6, color="#d0d0d0")
    ax.grid(which="minor", linestyle=":", linewidth=0.4, color="#e0e0e0", alpha=0.7)

    if any(lbl is not None for lbl in labels):
        ax.legend(
            fontsize=10,
            frameon=True,
            facecolor="white",
            edgecolor="#888888",
            shadow=True,
            fancybox=True,
        )


def _draw_trajectory(
    ax: plt.Axes,
    func: Callable[[np.ndarray], np.ndarray],
    paths: list[np.ndarray],
    labels: list[str | None],
    field_size: float,
    title: str,
) -> object:
    """Draw trajectory heatmap onto an existing Axes. Returns the contour for colorbar."""
    x = np.linspace(-field_size, field_size, 500)
    y = np.linspace(-field_size, field_size, 500)
    X, Y = np.meshgrid(x, y)
    Z = func(np.stack([X, Y], axis=-1))

    contour = ax.contourf(X, Y, Z, levels=60, cmap="viridis", alpha=0.9)
    ax.contour(X, Y, Z, levels=60, colors="black", linewidths=0.25, alpha=0.4)

    for i, (path, label) in enumerate(zip(paths, labels)):
        path = np.array(path)
        color = _TRAJ_COLORS[i % len(_TRAJ_COLORS)]

        if len(path) > 2:
            ax.plot(
                path[1:-1, 0],
                path[1:-1, 1],
                "o-",
                color=color,
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=color,
                linewidth=1.5,
                label=label,
            )

        deltas = np.diff(path, axis=0)
        ax.quiver(
            path[:-1, 0],
            path[:-1, 1],
            deltas[:, 0],
            deltas[:, 1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color=color,
            width=0.005,
            headwidth=4,
            alpha=0.8,
        )

        ax.plot(
            path[0, 0],
            path[0, 1],
            "o",
            color="white",
            markersize=8,
            zorder=5,
            label="Начальная точка" if i == 0 else None,
        )
        ax.plot(
            path[-1, 0],
            path[-1, 1],
            "*",
            color="yellow",
            markersize=12,
            zorder=5,
            label="Конечная точка" if i == 0 else None,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    legend_handles: list[Line2D] = []
    if any(lbl is not None for lbl in labels):
        for i, label in enumerate(labels):
            if label is not None:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=_TRAJ_COLORS[i % len(_TRAJ_COLORS)],
                        linewidth=2,
                        label=label,
                    )
                )
    legend_handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="white",
            markersize=8,
            label="Начальная точка",
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="yellow",
            markersize=12,
            label="Конечная точка",
            linestyle="",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=10,
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#555",
        labelcolor="white",
        fancybox=True,
        shadow=True,
    )

    return contour


def plot_convergence(
    func: Callable[[np.ndarray], np.ndarray],
    paths: np.ndarray | list[np.ndarray],
    labels: list[str | None] | None = None,
    yscale: Literal["log", "symlog", "linear"] | None = None,
    title: str = "Сходимость оптимизации",
) -> None:
    if isinstance(paths, np.ndarray):
        paths = [paths]
    if labels is None:
        labels = [None] * len(paths)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    _draw_convergence(ax, func, paths, labels, yscale, title)
    plt.tight_layout()
    plt.show()


def plot_trajectory(
    func: Callable[[np.ndarray], np.ndarray],
    paths: np.ndarray | list[np.ndarray],
    title: str = "Траектория оптимизации",
    field_size: float = 6,
    labels: list[str | None] | None = None,
) -> None:
    if isinstance(paths, np.ndarray):
        paths = [paths]
    if labels is None:
        labels = [None] * len(paths)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=110)
    contour = _draw_trajectory(ax, func, paths, labels, field_size, title)
    fig.colorbar(contour, ax=ax, label="f(x, y)")
    plt.tight_layout()
    plt.show()


def plot_combined(
    func: Callable[[np.ndarray], np.ndarray],
    paths: np.ndarray | list[np.ndarray],
    labels: list[str | None] | None = None,
    yscale: Literal["log", "symlog", "linear"] | None = None,
    conv_title: str = "Сходимость",
    traj_title: str = "Траектория",
    field_size: float = 6,
) -> None:
    """Plot convergence (left) and trajectory heatmap (right) side by side."""
    if isinstance(paths, np.ndarray):
        paths = [paths]
    if labels is None:
        labels = [None] * len(paths)

    fig, (ax_conv, ax_traj) = plt.subplots(1, 2, figsize=(20, 7), dpi=110)
    _draw_convergence(ax_conv, func, paths, labels, yscale, conv_title)
    contour = _draw_trajectory(ax_traj, func, paths, labels, field_size, traj_title)
    fig.colorbar(contour, ax=ax_traj, label="f(x, y)")
    plt.tight_layout()
    plt.show()
