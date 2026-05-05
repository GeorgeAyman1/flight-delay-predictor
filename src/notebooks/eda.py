"""
EDA Script — Flight Delay Predictor
=====================================
Dataset : train_selected.parquet  (processed features)
          train_preprocessed.parquet (with carrier_name / airport_name)
Target  : departure_delayed  (1 = delayed ≥15 min, 0 = on-time)

Usage
-----
    python eda.py                          # uses defaults
    DATA_DIR=/data/processed python eda.py # override data dir
    python eda.py --data-dir /data/processed --fig-dir /reports/figures
"""

import argparse
import os
import sys
from pathlib import Path
import seaborn as sns

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = ["#2196F3", "#F44336"]   # blue = on-time, red = delayed
BLUE    = "#2196F3"
RED     = "#F44336"
GRAY    = "#9E9E9E"


# ── CLI / env config ───────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flight delay EDA")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "../../data/processed"),
        help="Directory containing the parquet files (default: ../../data/processed)",
    )
    parser.add_argument(
        "--fig-dir",
        default=os.getenv("FIG_DIR", "../../reports/figures"),
        help="Output directory for figures (default: ../../reports/figures)",
    )
    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────
def save(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    """Save figure and close it — no plt.show() in pipeline mode."""
    path = fig_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def fmt_thousands(x, _):
    return f"{int(x):,}"


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_01_target_distribution(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 1 — Target distribution: bar + donut."""
    print("Plot 01: target distribution")
    counts = train["departure_delayed"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Distribution: Departure Delayed", fontsize=13, fontweight="bold")

    axes[0].bar(["On-Time", "Delayed"], counts.values,
                color=[BLUE, RED], width=0.4, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Flight Counts", fontweight="bold")
    axes[0].set_ylabel("Number of Flights")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_thousands))
    sns.despine(ax=axes[0])

    labels = [
        f"On-Time ({counts[0] / counts.sum() * 100:.1f}%)",
        f"Delayed ({counts[1] / counts.sum() * 100:.1f}%)",
    ]
    axes[1].pie(counts.values, labels=labels, colors=[BLUE, RED],
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 11})
    axes[1].set_title("Class Proportion", fontweight="bold")

    plt.tight_layout()
    save(fig, fig_dir, "01_target_distribution.png")


def plot_02_univariate_distributions(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 2 — KDE + histogram for key continuous features."""
    print("Plot 02: univariate distributions")
    features    = ["weather_severity", "tmpf", "cloud_ceiling"]
    feat_labels = ["Weather Severity Score", "Temperature (°F)", "Cloud Ceiling (ft)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Univariate Distributions of Key Continuous Features",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, feat, label in zip(axes, features, feat_labels):
        data = train[feat].dropna()
        # Sample for KDE to improve performance on large datasets
        data_for_kde = data.sample(min(50000, len(data)), random_state=42) if len(data) > 50000 else data
        
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1 / 3)) if iqr > 0 else 1
        n_bins = max(10, min(int((data.max() - data.min()) / bin_width), 80))

        ax.hist(data, bins=n_bins, color=BLUE, alpha=0.4, density=True, label="Histogram")
        data_for_kde.plot.kde(ax=ax, color=BLUE, linewidth=2.2, label="KDE")
        ax.axvline(data.mean(),   color=RED,      linestyle="--", linewidth=1.4,
                   label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="orange",  linestyle=":",  linewidth=1.4,
                   label=f"Median={data.median():.1f}")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    save(fig, fig_dir, "02_univariate_distributions.png")


def plot_03_weather_severity_by_delay(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 3 — Box + violin: weather severity by delay class."""
    print("Plot 03: weather severity box/violin")
    plot_df = train[["weather_severity", "departure_delayed"]].copy()
    plot_df["Status"] = plot_df["departure_delayed"].map({0: "On-Time", 1: "Delayed"})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Weather Severity vs Departure Delay Status",
                 fontsize=14, fontweight="bold")

    sns.boxplot(data=plot_df, x="Status", y="weather_severity",
                palette={"On-Time": BLUE, "Delayed": RED},
                order=["On-Time", "Delayed"], width=0.4,
                flierprops=dict(marker="o", markersize=2, alpha=0.3), ax=axes[0])
    axes[0].set_title("Box Plot", fontsize=12)
    axes[0].set_ylabel("Weather Severity Score")
    axes[0].set_xlabel("")
    sns.despine(ax=axes[0])

    sns.violinplot(data=plot_df, x="Status", y="weather_severity",
                   palette={"On-Time": BLUE, "Delayed": RED},
                   order=["On-Time", "Delayed"], inner="quartile", ax=axes[1])
    axes[1].set_title("Violin Plot (Box + KDE Shape)", fontsize=12)
    axes[1].set_ylabel("Weather Severity Score")
    axes[1].set_xlabel("")
    sns.despine(ax=axes[1])

    summary = plot_df.groupby("Status")["weather_severity"].agg(["mean", "median", "std"])
    print(summary.round(3))

    plt.tight_layout()
    save(fig, fig_dir, "03_weather_severity_by_delay.png")


def plot_04_weather_severity_delay_rate(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 4 — Delay rate per weather severity bin + flight volume."""
    print("Plot 04: weather severity → delay rate")
    plot_df = train[["weather_severity", "departure_delayed"]].copy()
    plot_df["severity_bin"] = pd.cut(
        plot_df["weather_severity"],
        bins=[0, 1, 2, 3, 4, 5, 7, 10, 25],
        labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-7", "7-10", "10+"],
    )

    bin_stats = (
        plot_df.groupby("severity_bin", observed=True)
        .agg(delay_rate=("departure_delayed", "mean"),
             count=("departure_delayed", "count"))
        .reset_index()
    )

    overall_rate = train["departure_delayed"].mean()
    bar_colors = [RED if r > overall_rate else BLUE for r in bin_stats["delay_rate"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("How Weather Severity Drives Delay Rate", fontsize=14, fontweight="bold")

    bars = axes[0].bar(bin_stats["severity_bin"].astype(str),
                       bin_stats["delay_rate"] * 100,
                       color=bar_colors, width=0.6, edgecolor="white", linewidth=1.2)
    for bar, rate, count in zip(bars, bin_stats["delay_rate"], bin_stats["count"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f"{rate*100:.1f}%\nn={count:,}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    axes[0].axhline(overall_rate * 100, color=GRAY, linestyle="--", linewidth=1.4,
                    label=f"Overall avg: {overall_rate*100:.1f}%")
    axes[0].set_title("Delay Rate Increases with Severity", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Delay Rate (%)", fontsize=11)
    axes[0].set_xlabel("Weather Severity Score Bin", fontsize=11)
    axes[0].legend(fontsize=9)
    sns.despine(ax=axes[0])

    axes[1].bar(bin_stats["severity_bin"].astype(str), bin_stats["count"],
                color=BLUE, alpha=0.7, width=0.6, edgecolor="white", linewidth=1.2)
    for i, (count, _) in enumerate(zip(bin_stats["count"], bin_stats["delay_rate"])):
        axes[1].text(i, count + bin_stats["count"].max() * 0.01,
                     f"{count:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axes[1].set_title("Flight Volume per Severity Bin", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Number of Flights", fontsize=11)
    axes[1].set_xlabel("Weather Severity Score Bin", fontsize=11)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_thousands))
    sns.despine(ax=axes[1])

    plt.tight_layout()
    save(fig, fig_dir, "03_weather_severity_delay_rate.png")
    print(bin_stats[["severity_bin", "delay_rate", "count"]].to_string(index=False))


def plot_05_severe_weather_only(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 5 — Box plot for severe weather conditions only (severity >= 12)."""
    print("Plot 05: severe weather subset")
    severe_df = train[train["weather_severity"] >= 12].copy()
    severe_df["Status"] = severe_df["departure_delayed"].map({0: "On-Time", 1: "Delayed"})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=severe_df, x="Status", y="weather_severity",
                palette={"On-Time": BLUE, "Delayed": RED}, ax=ax)
    ax.set_title("Weather Severity Distribution (Severe Conditions Only)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Weather Severity Score")
    sns.despine(ax=ax)

    plt.tight_layout()
    save(fig, fig_dir, "05_severe_weather_box.png")


def plot_06_delay_rate_by_tod(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 6 — Clustered bar: flight counts by time of day."""
    print("Plot 06: delay rate by time of day")
    tod_cols = ["tod_late_night", "tod_early_morning", "tod_afternoon", "tod_evening"]
    tod_labels = {
        "tod_late_night":    "Late Night\n(23:00–04:59)",
        "tod_early_morning": "Early Morning\n(05:00–11:59)",
        "tod_afternoon":     "Afternoon\n(12:00–16:59)",
        "tod_evening":       "Evening\n(17:00–22:59)",
    }
    available_tod = [c for c in tod_cols if c in train.columns]

    # Use vectorized operations instead of apply for performance
    plot_df = train[available_tod + ["departure_delayed"]].copy()
    # Initialize with default, then set based on which column is 1
    time_of_day = pd.Series(["tod_late_night"] * len(plot_df), index=plot_df.index)
    for col in available_tod:
        time_of_day = time_of_day.where(plot_df[col] != 1, col)
    plot_df["time_of_day"] = time_of_day
    plot_df["Status"]      = plot_df["departure_delayed"].map({0: "On-Time", 1: "Delayed"})
    plot_df["TOD_Label"]   = plot_df["time_of_day"].map(tod_labels)

    tod_order = [
        "Late Night\n(23:00–04:59)",
        "Early Morning\n(05:00–11:59)",
        "Afternoon\n(12:00–16:59)",
        "Evening\n(17:00–22:59)",
    ]

    counts_df = (
        plot_df.groupby(["TOD_Label", "Status"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=counts_df, x="TOD_Label", y="count",
                hue="Status", hue_order=["On-Time", "Delayed"],
                palette={"On-Time": BLUE, "Delayed": RED},
                order=tod_order, ax=ax)

    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 500,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Flight Counts by Time of Day and Departure Status",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Flights", fontsize=12)
    ax.set_xlabel("Scheduled Departure Window", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_thousands))
    ax.legend(title="Departure Status", fontsize=10, title_fontsize=10)
    sns.despine()

    plt.tight_layout()
    save(fig, fig_dir, "04_clustered_bar_time_of_day.png")


def plot_07_delay_rate_by_dow(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 7 — Delay rate by day of week + per-day histograms."""
    print("Plot 07: delay rate by day of week")
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    dow_stats = (
        train.groupby("day_of_week")
        .agg(delay_rate=("departure_delayed", "mean"),
             count=("departure_delayed", "count"))
        .reset_index()
    )
    dow_stats["day_name"] = dow_stats["day_of_week"].map(day_map)
    overall_rate = train["departure_delayed"].mean() * 100

    # — Bar chart —
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = [RED if r > overall_rate / 100 else BLUE for r in dow_stats["delay_rate"]]
    bars = ax.bar(dow_stats["day_name"], dow_stats["delay_rate"] * 100,
                  color=bar_colors, width=0.55, edgecolor="white", linewidth=1.2)
    for bar, rate in zip(bars, dow_stats["delay_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{rate*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(overall_rate, color=GRAY, linestyle="--", linewidth=1.4)
    ax.set_title("Departure Delay Rate by Day of Week", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Delay Rate (%)", fontsize=12)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylim(0, dow_stats["delay_rate"].max() * 130)
    legend_elements = [
        Patch(facecolor=RED,  label="Above average"),
        Patch(facecolor=BLUE, label="Below average"),
        plt.Line2D([0], [0], color=GRAY, linestyle="--", label=f"Avg: {overall_rate:.1f}%"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    sns.despine()
    plt.tight_layout()
    save(fig, fig_dir, "05_delay_rate_by_day_of_week.png")
    print(dow_stats[["day_name", "delay_rate", "count"]].to_string(index=False))

    # — Per-day histograms —
    fig2, axes2 = plt.subplots(1, 7, figsize=(20, 4), sharey=True)
    fig2.suptitle("Delay Distribution per Day of Week — Histograms",
                  fontsize=13, fontweight="bold")
    for j, (_, row) in enumerate(dow_stats.iterrows()):
        subset = train[train["day_of_week"] == row["day_of_week"]]
        for val, color, label in [(0, BLUE, "On-Time"), (1, RED, "Delayed")]:
            axes2[j].hist(subset[subset["departure_delayed"] == val]["departure_delayed"],
                          bins=3, color=color, alpha=0.6, label=label, edgecolor="white")
        axes2[j].set_title(f"{row['day_name']}\n{row['delay_rate']*100:.1f}%",
                           fontsize=9, fontweight="bold")
        axes2[j].set_xlabel("0=On-Time | 1=Delayed")
        axes2[j].set_ylabel("Count" if j == 0 else "")
        axes2[j].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_thousands))
        if j == 0:
            axes2[j].legend(fontsize=7)
        sns.despine(ax=axes2[j])
    plt.tight_layout()
    save(fig2, fig_dir, "05b_dow_histograms.png")


def plot_08_cascade_effect(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 8 — Cascade effect: previous flight delay → current delay."""
    print("Plot 08: cascade effect")
    overall_rate = train["departure_delayed"].mean() * 100

    delay_by_prev = train.groupby("prev_flight_delayed")["departure_delayed"].mean() * 100
    delay_by_prev.index = ["Prev Flight\nOn-Time", "Prev Flight\nDelayed"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(delay_by_prev.index, delay_by_prev.values,
                  color=[BLUE, RED], width=0.4, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, delay_by_prev.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.axhline(overall_rate, color=GRAY, linestyle="--", linewidth=1.3,
               label=f"Overall avg: {overall_rate:.1f}%")
    ax.set_title("Delay Rate by Previous Flight Status", fontsize=11)
    ax.set_ylabel("Delay Rate (%)")
    ax.set_ylim(0, delay_by_prev.max() * 1.25)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    save(fig, fig_dir, "06_delay_by_previous_flight.png")

    # Stacked bar + contingency table
    counts_table = pd.crosstab(train["prev_flight_delayed"], train["departure_delayed"])
    counts_table.index   = ["Prev On-Time (0)", "Prev Delayed (1)"]
    counts_table.columns = ["On-Time", "Delayed"]

    crosstab = pd.crosstab(train["prev_flight_delayed"], train["departure_delayed"],
                           normalize="index") * 100
    crosstab.index   = ["Prev On-Time (0)", "Prev Delayed (1)"]
    crosstab.columns = ["On-Time", "Delayed"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cascade Effect: Previous vs Current Flight Delay",
                 fontsize=13, fontweight="bold")

    crosstab.plot(kind="bar", stacked=True, ax=axes[0],
                  color=[BLUE, RED], edgecolor="white", linewidth=1.2, width=0.4)
    axes[0].set_title("100% Stacked Composition", fontsize=11)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xlabel("Previous Flight Status")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(["On-Time", "Delayed"], fontsize=9)
    sns.despine(ax=axes[0])

    axes[1].axis("off")
    table = axes[1].table(cellText=counts_table.values,
                          rowLabels=counts_table.index,
                          colLabels=counts_table.columns,
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1].set_title("Contingency Table (Counts)", fontsize=11)

    plt.tight_layout()
    save(fig, fig_dir, "06c_stacked_with_table.png")


def plot_09_correlation_heatmap(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 9 — Correlation heatmap sorted by |corr with target|."""
    print("Plot 09: correlation heatmap")
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    corr = train[numeric_cols].corr()
    target_corr_order = corr["departure_delayed"].abs().sort_values(ascending=False).index
    corr_sorted = corr.loc[target_corr_order, target_corr_order]
    mask = np.triu(np.ones_like(corr_sorted, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr_sorted, mask=mask, annot=True, fmt=".2f",
                annot_kws={"size": 8}, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax)
    ax.set_title("Feature Correlation Matrix\n(sorted by |correlation with target|)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    save(fig, fig_dir, "07_correlation_heatmap.png")

    print("Top correlations with departure_delayed:")
    print(corr["departure_delayed"].drop("departure_delayed").abs()
          .sort_values(ascending=False).round(4))


def plot_10_airline_delay_rate(train_pre: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 10 — Airline delay rate bar chart."""
    print("Plot 10: airline delay rate")
    airline_rates = (
        train_pre.groupby("carrier_name")["departure_delayed"]
        .agg(delay_rate="mean", count="size")
        .reset_index()
        .sort_values("delay_rate")
    )
    overall_avg = train_pre["departure_delayed"].mean()
    bar_colors  = [RED if r > overall_avg else BLUE for r in airline_rates["delay_rate"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(airline_rates["carrier_name"], airline_rates["delay_rate"] * 100,
                  color=bar_colors, width=0.6, edgecolor="white", linewidth=1.2)
    for bar, rate, count in zip(bars, airline_rates["delay_rate"], airline_rates["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{rate*100:.1f}%\nn={count:,}", ha="center", va="bottom", fontsize=9)
    ax.axhline(overall_avg * 100, color=GRAY, linestyle="--", linewidth=1.4,
               label=f"Overall avg: {overall_avg*100:.1f}%")
    ax.set_title("Airline Delay Rate Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Delay Rate (%)")
    ax.set_xlabel("Airline")
    ax.set_xticklabels(airline_rates["carrier_name"], rotation=45, ha="right")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    save(fig, fig_dir, "airline_delay_rate_fixed.png")


def plot_11_airport_delay_rate(train_pre: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 11 — Airport delay rate horizontal bar chart."""
    print("Plot 11: airport delay rate")
    airport_rates = (
        train_pre.groupby("airport_name")["departure_delayed"]
        .agg(delay_rate="mean", count="size")
        .reset_index()
        .sort_values("delay_rate", ascending=True)
    )
    overall_avg = train_pre["departure_delayed"].mean()
    bar_colors  = [RED if r > overall_avg else BLUE for r in airport_rates["delay_rate"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(airport_rates["airport_name"], airport_rates["delay_rate"] * 100,
                   color=bar_colors, height=0.6, edgecolor="white", linewidth=1.2)
    for bar, rate in zip(bars, airport_rates["delay_rate"]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{rate*100:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.axvline(overall_avg * 100, color=GRAY, linestyle="--", linewidth=1.4,
               label=f"Overall avg: {overall_avg*100:.1f}%")
    ax.set_title("Airport Delay Rate Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Delay Rate (%)")
    ax.set_ylabel("Airport")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    save(fig, fig_dir, "airport_delay_rate_fixed.png")


def plot_12_route_delay_rate(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 12 — Route delay rate KDE + risk-zone bar chart."""
    print("Plot 12: route delay rate distribution")
    route_rates  = train.drop_duplicates("route_delay_rate")["route_delay_rate"].dropna()
    overall_mean = train["departure_delayed"].mean()

    q75, q25 = np.percentile(route_rates, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(route_rates) ** (1 / 3)) if iqr > 0 else 0.01
    n_bins = max(20, min(int((route_rates.max() - route_rates.min()) / bin_width), 60))

    risk_bins   = [0, 0.15, 0.20, 0.25, 0.30, 1.0]
    risk_labels = ["Very Low\n(<15%)", "Low\n(15–20%)",
                   "Average\n(20–25%)", "High\n(25–30%)", "Very High\n(>30%)"]
    risk_colors = [BLUE, BLUE, GRAY, RED, RED]
    route_risk  = pd.cut(route_rates, bins=risk_bins, labels=risk_labels, include_lowest=True)
    risk_counts = route_risk.value_counts().reindex(risk_labels)
    risk_pct    = risk_counts / risk_counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Route Delay Rate Distribution\n(~1,093 Unique Routes)",
                 fontsize=14, fontweight="bold")

    # Sample for KDE if needed
    route_rates_for_kde = route_rates.sample(min(1000, len(route_rates)), random_state=42) if len(route_rates) > 1000 else route_rates
    
    axes[0].hist(route_rates, bins=n_bins, color=BLUE, alpha=0.35,
                 density=True, label="Histogram (unique routes)")
    route_rates_for_kde.plot.kde(ax=axes[0], color=BLUE, linewidth=2.2, label="KDE")
    axes[0].axvline(overall_mean, color=RED, linestyle="--", linewidth=1.6,
                    label=f"Overall avg: {overall_mean*100:.1f}%")
    axes[0].axvline(route_rates.mean(), color="orange", linestyle=":", linewidth=1.6,
                    label=f"Route mean: {route_rates.mean()*100:.1f}%")
    axes[0].set_xlim(0, route_rates.quantile(0.995))
    axes[0].set_title("Unique Route Rates", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Historical Delay Rate", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    axes[0].legend(fontsize=8)
    sns.despine(ax=axes[0])

    bars = axes[1].bar(risk_labels, risk_counts.values, color=risk_colors,
                       width=0.55, edgecolor="white", linewidth=1.2)
    for bar, count, pct in zip(bars, risk_counts.values, risk_pct.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{count:,} routes\n({pct:.1f}%)",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_title("Routes by Delay Risk Zone", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Number of Routes", fontsize=11)
    axes[1].set_xlabel("Delay Risk Zone", fontsize=11)
    axes[1].set_ylim(0, risk_counts.max() * 1.25)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    save(fig, fig_dir, "extra_route_delay_rate_distribution.png")

    print(f"\nRoutes ABOVE overall average ({overall_mean*100:.1f}%): "
          f"{(route_rates > overall_mean).sum():,} "
          f"({(route_rates > overall_mean).mean()*100:.1f}%)")
    print(f"Routes BELOW overall average: "
          f"{(route_rates <= overall_mean).sum():,} "
          f"({(route_rates <= overall_mean).mean()*100:.1f}%)")


def plot_13_route_congestion(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 13 — Route congestion vs delay status."""
    print("Plot 13: route congestion")
    congestion_map  = {0: "Low", 1: "Medium", 2: "High"}
    congestion_order = ["Low", "Medium", "High"]

    plot_df = train[["route_congestion", "departure_delayed"]].copy()
    plot_df["Congestion"] = plot_df["route_congestion"].map(congestion_map)
    plot_df["Status"]     = plot_df["departure_delayed"].map({0: "On-Time", 1: "Delayed"})

    counts_df = (
        plot_df.groupby(["Congestion", "Status"])
        .size()
        .reset_index(name="count")
    )
    crosstab = pd.crosstab(plot_df["Congestion"], plot_df["Status"],
                           normalize="index") * 100
    crosstab = crosstab.reindex(congestion_order)[["On-Time", "Delayed"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Route Congestion vs Departure Delay Status",
                 fontsize=14, fontweight="bold")

    sns.barplot(data=counts_df, x="Congestion", y="count",
                hue="Status", hue_order=["On-Time", "Delayed"],
                palette={"On-Time": BLUE, "Delayed": RED},
                order=congestion_order, ax=axes[0])
    for bar in axes[0].patches:
        h = bar.get_height()
        if h > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1000,
                         f"{int(h):,}", ha="center", va="bottom",
                         fontsize=8, fontweight="bold")
    axes[0].set_title("Clustered Bar — Flight Counts", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Flights")
    axes[0].set_xlabel("Route Congestion Level")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_thousands))
    axes[0].legend(title="Status", fontsize=9)
    sns.despine(ax=axes[0])

    crosstab.plot(kind="bar", stacked=True, color=[BLUE, RED],
                  edgecolor="white", linewidth=1.2, width=0.5, ax=axes[1])
    for i, (_, row) in enumerate(crosstab.iterrows()):
        axes[1].text(i, row["On-Time"] / 2, f"{row['On-Time']:.1f}%",
                     ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        axes[1].text(i, row["On-Time"] + row["Delayed"] / 2, f"{row['Delayed']:.1f}%",
                     ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    axes[1].set_title("100% Stacked Bar — Delay Rate", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_xlabel("Route Congestion Level")
    axes[1].set_xticklabels(congestion_order, rotation=0)
    axes[1].legend(["On-Time", "Delayed"], loc="upper right", fontsize=9)
    axes[1].set_ylim(0, 100)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    save(fig, fig_dir, "extra_route_congestion.png")
    print("\nDelay rate by congestion level:")
    print(crosstab["Delayed"].round(2))


def plot_14_heatmap_dow_tod(train: pd.DataFrame, fig_dir: Path) -> None:
    """Plot 14 — Delay rate heatmap: day of week × time of day."""
    print("Plot 14: heatmap day-of-week × time-of-day")
    tod_map = {
        "tod_early_morning": "Early Morning",
        "tod_afternoon":     "Afternoon",
        "tod_evening":       "Evening",
    }

    # Vectorized reconstruction instead of apply for performance
    tod_series = pd.Series("Morning", index=train.index)
    for col, label in tod_map.items():
        if col in train.columns:
            tod_series = tod_series.where(train[col] != 1, label)
    
    dow_series = train["day_of_week"].map(
        {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    )

    pivot_input = pd.DataFrame({
        "dow_label":       dow_series,
        "time_of_day":     tod_series,
        "departure_delayed": train["departure_delayed"],
    })

    TOD_ORDER = ["Early Morning", "Morning", "Afternoon", "Evening"]
    DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    pivot = (
        pivot_input
        .groupby(["dow_label", "time_of_day"])["departure_delayed"]
        .mean()
        .mul(100)
        .reset_index()
        .pivot(index="dow_label", columns="time_of_day", values="departure_delayed")
        .reindex(index=DOW_ORDER, columns=TOD_ORDER)
        .astype(float)
    )
    print(pivot.round(1))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", annot_kws={"size": 8},
                cmap="RdYlGn_r", linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Delay rate (%)"})
    ax.set_title("Delay Rate (%) by Day of Week × Time of Day",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Time of day", labelpad=8)
    ax.set_ylabel("Day of week", labelpad=8)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    sns.despine(ax=ax)

    plt.tight_layout()
    save(fig, fig_dir, "eda_11_delay_heatmap_dow_tod.png")

    stacked = pivot.stack()
    print("\nHighest-risk slots:")
    print(stacked.sort_values(ascending=False).head(5).round(1).to_string())
    print("\nLowest-risk slots:")
    print(stacked.sort_values().head(5).round(1).to_string())


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args    = parse_args()
    data_dir = Path(args.data_dir)
    fig_dir  = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir : {data_dir}")
    print(f"Figure dir: {fig_dir}")

    # Load data
    print("\nLoading data...")
    train = pd.read_parquet(data_dir / "train_selected.parquet", engine="pyarrow")
    train_pre = pd.read_parquet(data_dir / "train_preprocessed.parquet", engine="pyarrow")
    print(f"  train     shape: {train.shape}")
    print(f"  train_pre shape: {train_pre.shape}")

    # Run all plots
    print("\nGenerating figures...")
    plot_01_target_distribution(train, fig_dir)
    plot_02_univariate_distributions(train, fig_dir)
    plot_03_weather_severity_by_delay(train, fig_dir)
    plot_04_weather_severity_delay_rate(train, fig_dir)
    plot_05_severe_weather_only(train, fig_dir)
    plot_06_delay_rate_by_tod(train, fig_dir)
    plot_07_delay_rate_by_dow(train, fig_dir)
    plot_08_cascade_effect(train, fig_dir)
    plot_09_correlation_heatmap(train, fig_dir)
    plot_10_airline_delay_rate(train_pre, fig_dir)
    plot_11_airport_delay_rate(train_pre, fig_dir)
    plot_12_route_delay_rate(train, fig_dir)
    plot_13_route_congestion(train, fig_dir)
    plot_14_heatmap_dow_tod(train, fig_dir)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
