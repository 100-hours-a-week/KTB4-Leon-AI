from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import seaborn as sns


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": {"light": "#CEDFFE", "base": "#A3BEFA", "dark": "#2E4780"},
    "gold": {"light": "#FFEA8F", "base": "#FFE15B", "dark": "#736422"},
    "orange": {"light": "#FFBDA1", "base": "#F0986E", "dark": "#804126"},
    "olive": {"light": "#BEEB96", "base": "#A3D576", "dark": "#386411"},
}


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Apple SD Gothic Neo",
                "Noto Sans CJK KR",
                "DejaVu Sans",
                "Arial",
            ],
        },
    )


def add_chart_header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
) -> None:
    title = textwrap.fill(title, width=72, break_long_words=False)
    subtitle = textwrap.fill(subtitle, width=100, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1

    fig.subplots_adjust(
        top=max(0.66, 0.88 - 0.045 * (title_lines - 1) - 0.03 * (subtitle_lines - 1))
    )
    left = ax.get_position().x0
    fig.text(
        left,
        0.98,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925 - 0.045 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )


def finish_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"], labelsize=9)

