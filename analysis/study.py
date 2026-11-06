"""Every table and figure of the final study, from ``results/`` (Phases 7-9).

Tables (``results/tables/``): ``main`` (all methods x scenarios x demands, with the
D33 verdict), ``training`` (budget, time, plateau), ``ablation`` (reward and
observation type, switch penalty), ``robustness``, ``scalability``, ``transfer``,
``federated``. Figures (``results/figures/``): ``curves_<scenario>``,
``metrics_<scenario>``, ``scalability``, ``robustness``, ``federated`` and
``heatmap_<scenario>_<profile>`` (from experiments/heatmap.py). Each part is skipped
when its runs do not exist yet, so the script also works on a partial sweep.

Usage:
    python analysis/study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_table import METHOD_ORDER, PROFILE_ORDER, build, per_seed  # noqa: E402
from baseline_table import to_markdown as main_markdown  # noqa: E402
from marl_study import evaluation_metrics, learning_curves, training_table  # noqa: E402

from trafficmarl.metrics.stats import significantly_lower, summarize, welch_p  # noqa: E402
from trafficmarl.utils import paths  # noqa: E402

PREFIX = "marl"
SCENARIOS = ["single", "corridor", "grid3x3", "grid5x5", "lust"]
INTERSECTIONS = {"single": 1, "corridor": 4, "grid3x3": 9, "lust": 13, "grid5x5": 25}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3dc"
# Colour follows the family of methods; each row or line is also labelled by name.
FAMILY = {
    "traditional": "#52514e",
    "centralized": "#eda100",
    "independent": "#2a78d6",
    "implicit": "#eb6834",
    "explicit": "#1baf7a",
    "CTDE": "#e87ba4",
}
METHODS = {
    "fixed_time": ("Fixed time (SUMO)", "traditional"),
    "webster": ("Webster", "traditional"),
    "actuated": ("Actuated", "traditional"),
    "max_pressure": ("Max-pressure", "traditional"),
    "max_pressure_maxred": ("Max-pressure, max red", "traditional"),
    "centralized_ppo": ("Centralized PPO", "centralized"),
    "ippo_local": ("IPPO, local reward", "independent"),
    "ippo_neighborhood": ("IPPO, neighbourhood reward", "implicit"),
    "ippo_global": ("IPPO, global reward", "implicit"),
    "ippo_extended": ("IPPO, neighbour observation", "explicit"),
    "mappo": ("MAPPO", "CTDE"),
    "qmix": ("QMIX", "CTDE"),
    "maddpg": ("MADDPG", "CTDE"),
}
FEDERATED = {
    "ippo_local": ("Parameter sharing (IPPO)", "#2a78d6"),
    "independent": ("Independent policies", "#52514e"),
    "fedavg_r1": ("FedAvg, every iteration", "#eb6834"),
    "fedavg_r5": ("FedAvg, every 5", "#1baf7a"),
    "fedavg_r20": ("FedAvg, every 20", "#eda100"),
    "fedavg_r5_dp": ("FedAvg, every 5, noise", "#e87ba4"),
}
LOCAL_OBS_DIM = 47  # floats in a local observation, identical in every scenario (D20)
FLOAT_BYTES = 4


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    out = paths.RESULTS_DIR / "figures" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def _write(name: str, text: str) -> None:
    out = paths.RESULTS_DIR / "tables" / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


def _present(scenarios=SCENARIOS) -> list[str]:
    return [s for s in scenarios if any((paths.RESULTS_DIR / f"{PREFIX}_{s}").glob("eval_*/*/*"))]


def _seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    return per_seed(metrics[metrics.method.isin(METHOD_ORDER)])


def main_tables(metrics: pd.DataFrame) -> pd.DataFrame:
    main = metrics[metrics.profile.isin(PROFILE_ORDER)]
    table = build(_seeds(main))
    table.to_csv(paths.RESULTS_DIR / "tables" / "main.csv", index=False)
    _write("main", main_markdown(table))
    return table


def metric_figure(table: pd.DataFrame, scenario: str, profile: str = "medium") -> None:
    """One panel per metric, one row per method. Throughput shows no vehicle was held back."""
    data = table[(table.scenario == scenario) & (table.profile == profile)]
    data = data[data.method.isin(METHODS)]
    if data.empty:
        return
    data = data.iloc[::-1]
    panels = {
        "mean_waiting_time": "Waiting time per vehicle (s)",
        "network_queue_mean": "Queue length (vehicles)",
        "throughput": "Throughput (vehicles)",
        "co2_total_kg": "CO₂ (kg)",
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 0.34 * len(data) + 1.3), sharey=True,
                             constrained_layout=True)  # fmt: skip
    y = np.arange(len(data))
    for ax, (metric, title) in zip(axes, panels.items(), strict=True):
        colors = [FAMILY[METHODS[m][1]] for m in data.method]
        ax.errorbar(data[metric], y, xerr=data[f"{metric}_ci"].fillna(0), fmt="none",
                    ecolor=MUTED, elinewidth=1, zorder=2)  # fmt: skip
        ax.scatter(data[metric], y, c=colors, s=60, zorder=3, edgecolors="#fcfcfb", linewidths=1.5)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        _clean(ax)
    axes[0].set_yticks(y, [METHODS[m][0] for m in data.method])
    axes[0].tick_params(axis="y", length=0, labelsize=9.5, labelcolor=INK)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f, markersize=7)
               for f, c in FAMILY.items() if f in {METHODS[m][1] for m in data.method}]  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False,
               fontsize=9)  # fmt: skip
    fig.suptitle(f"{scenario}, {profile} demand: mean and 95 % CI over seeds", x=0.01, ha="left",
                 fontsize=11, color=INK)  # fmt: skip
    _save(fig, f"metrics_{scenario}")


def curves(scenarios: list[str]) -> None:
    variants = {m: (label, FAMILY[family]) for m, (label, family) in METHODS.items()}
    distinct = {  # learning curves overlap: here every method needs its own colour
        "centralized_ppo": "#eda100", "ippo_local": "#2a78d6", "ippo_neighborhood": "#eb6834",
        "ippo_global": "#6250d6", "ippo_extended": "#1baf7a", "mappo": "#e87ba4",
        "qmix": "#008300", "maddpg": "#52514e",
    }  # fmt: skip
    variants = {m: (variants[m][0], c) for m, c in distinct.items()}
    for scenario in scenarios:
        learning_curves(
            paths.RESULTS_DIR / "figures" / f"curves_{scenario}.png", PREFIX, [scenario], variants
        )


def ablation(seeds: pd.DataFrame) -> None:
    """Reward type (same observation) and observation type (same reward), against ippo_local."""
    methods = ["ippo_local", "ippo_neighborhood", "ippo_global", "ippo_extended"]
    lines = ["## Reward and observation type (IPPO, parameter sharing)\n",
             "| Scenario | Demand | Variant | n | Waiting (s) | Δ vs local | p (Welch) "
             "| Significantly lower | p95 delay (s) | Throughput |", "|" + "---|" * 10]  # fmt: skip
    for scenario in SCENARIOS:
        for profile in PROFILE_ORDER:
            group = seeds[(seeds.scenario == scenario) & (seeds.profile == profile)]
            ref = group[group.method == "ippo_local"]
            for method in methods:
                m = group[group.method == method]
                if m.empty or ref.empty or scenario == "single":
                    continue
                w, d = summarize(m.mean_waiting_time), summarize(m.p95_total_delay)
                same = method == "ippo_local"
                delta = (
                    "" if same else f"{100 * (w.mean / ref.mean_waiting_time.mean() - 1):+.1f} %"
                )
                p = "" if same else f"{welch_p(m.mean_waiting_time, ref.mean_waiting_time):.3f}"
                low = significantly_lower(m.mean_waiting_time, ref.mean_waiting_time)
                verdict = "reference" if same else ("yes" if low else "no")
                lines.append(
                    f"| {scenario} | {profile} | {METHODS[method][0]} | {len(m)} "
                    f"| {w.mean:.1f} ± {w.ci_half_width:.1f} | {delta} | {p} | {verdict} "
                    f"| {d.mean:.1f} ± {d.ci_half_width:.1f} | {m.throughput.mean():.0f} |"
                )
    root = paths.RESULTS_DIR / f"{PREFIX}_grid3x3"
    penalty = [("ippo_c01", 0.1), ("ippo_local", 0.5), ("ippo_c10", 1.0)]
    rows = []
    for method, c in penalty:
        files = list(root.glob(f"eval_medium/{method}/*/metrics.csv"))
        if files:
            t = pd.concat(pd.read_csv(f) for f in files).groupby("seed").mean(numeric_only=True)
            w = summarize(t.mean_waiting_time)
            rows.append(f"| {c} | {len(t)} | {w.mean:.1f} ± {w.ci_half_width:.1f} "
                        f"| {t.phase_changes_per_signal.mean():.0f} "
                        f"| {t.throughput.mean():.0f} |")  # fmt: skip
    if rows:
        lines += ["\n## Switch penalty c (IPPO, grid3x3, medium demand)\n",
                  "| c | n | Waiting (s) | Phase changes per signal | Throughput |",
                  "|---|---|---|---|---|", *rows]  # fmt: skip
    _write("ablation", "\n".join(lines))


def robustness(metrics: pd.DataFrame, scenarios: list[str]) -> None:
    """Policies trained on medium demand, tested on unseen demand levels."""
    seeds = _seeds(metrics)
    levels = ["medium", "high", "peak", "saturated"]
    colors = dict(zip(levels, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
    lines = ["| Scenario | Method | " + " | ".join(levels) + " |", "|" + "---|" * 6]
    panels = [s for s in scenarios if s in ("corridor", "grid3x3", "grid5x5", "lust")]
    fig, axes = plt.subplots(1, max(len(panels), 1), figsize=(4.2 * max(len(panels), 1), 5.2),
                             sharey=True, constrained_layout=True, squeeze=False)  # fmt: skip
    order = [m for m in METHODS if m != "webster"]
    for ax, scenario in zip(axes[0], panels, strict=False):
        group = seeds[seeds.scenario == scenario]
        y = np.arange(len(order))[::-1]
        for level in levels:
            data = group[group.profile == level].groupby("method").mean_waiting_time.mean()
            xs = [data.get(m, np.nan) for m in order]
            ax.scatter(xs, y, s=42, color=colors[level], label=level, zorder=3,
                       edgecolors="#fcfcfb", linewidths=1)  # fmt: skip
        ax.set_xscale("log")
        ax.set_title(scenario, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("waiting time per vehicle (s, log scale)", fontsize=9, color=MUTED)
        _clean(ax)
        ax.set_yticks(y, [METHODS[m][0] for m in order])
        for method in order:
            cells = []
            for level in levels:
                values = group[(group.profile == level) & (group.method == method)]
                s = summarize(values.mean_waiting_time) if len(values) else None
                ci = f" ± {s.ci_half_width:.1f}" if s and s.ci_half_width == s.ci_half_width else ""
                cells.append(f"{s.mean:.1f}{ci}" if s else "")
            if any(cells):
                lines.append(f"| {scenario} | {METHODS[method][0]} | " + " | ".join(cells) + " |")
    handles = axes[0][0].get_legend_handles_labels()
    if handles[0]:
        fig.legend(*handles, loc="outside lower center", ncols=4, frameon=False, fontsize=9,
                   title="demand (training: medium)")  # fmt: skip
    _save(fig, "robustness")
    _write("robustness", "Mean waiting time (s) by test demand; training on medium only.\n\n"
           + "\n".join(lines))  # fmt: skip


def scalability(table: pd.DataFrame, scenarios: list[str]) -> None:
    """Performance relative to Webster (D11) and training cost against the network size."""
    data = table[table.profile == "medium"]
    shown = ["actuated", "max_pressure", "centralized_ppo", "ippo_local", "ippo_extended",
             "mappo", "qmix", "maddpg"]  # fmt: skip
    colors = {"actuated": "#52514e", "max_pressure": "#008300", "centralized_ppo": "#eda100",
              "ippo_local": "#2a78d6", "ippo_extended": "#1baf7a", "mappo": "#e87ba4",
              "qmix": "#6250d6", "maddpg": "#eb6834"}  # fmt: skip
    rows = []
    for scenario in scenarios:
        group = data[data.scenario == scenario]
        ref = group[group.method == "webster"].mean_waiting_time
        for method in shown:
            m = group[group.method == method]
            if m.empty or ref.empty:
                continue
            stats = _training_stats(scenario, method)
            rows.append({
                "scenario": scenario, "n": INTERSECTIONS[scenario], "method": method,
                "seeds": int(m.n_seeds.iloc[0]),
                "improvement": 100 * (1 - m.mean_waiting_time.iloc[0] / ref.iloc[0]),
                "waiting": m.mean_waiting_time.iloc[0],
                **stats,
            })  # fmt: skip
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    panels = [("improvement", "Waiting time improvement vs Webster (%)"),
              ("wall_min", "Training time per run (min)"),
              ("steps_per_s", "Training speed (environment steps/s)")]  # fmt: skip
    for ax, (column, title) in zip(axes, panels, strict=True):
        for method in shown:
            m = frame[frame.method == method].dropna(subset=[column]).sort_values("n")
            if len(m):
                ax.plot(m.n, m[column], "-o", color=colors[method], linewidth=2, markersize=6,
                        label=METHODS[method][0])  # fmt: skip
        ax.set_xscale("log")
        ax.set_xticks([1, 4, 9, 13, 25], ["1", "4", "9", "13\n(LuST)", "25"])
        ax.minorticks_off()
        ax.set_xlabel("intersections", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=8, frameon=False, fontsize=9)
    _save(fig, "scalability")
    lines = ["| Scenario | Intersections | Method | n | Waiting (s) | vs Webster "
             "| Train time (min) | Steps/s | Peak memory (MB) |", "|" + "---|" * 9]  # fmt: skip
    for r in frame.sort_values(["n", "method"]).itertuples():
        lines.append(
            f"| {r.scenario} | {r.n} | {METHODS[r.method][0]} | {r.seeds} | {r.waiting:.1f} "
            f"| {r.improvement:+.0f} % | {r.wall_min:.0f} | {r.steps_per_s:.0f} | {r.memory} |"
        )
    _write("scalability", "\n".join(lines))


def _training_stats(scenario: str, method: str) -> dict:
    files = list(
        (paths.RESULTS_DIR / f"{PREFIX}_{scenario}" / method).glob("*/training_stats.json")
    )
    if not files and method == "centralized_ppo":
        root = paths.RESULTS_DIR / "baselines" / "train_centralized_ppo" / scenario
        files = list(root.glob("*/training_stats.json"))
    if not files:
        return {"wall_min": np.nan, "steps_per_s": np.nan, "memory": ""}
    stats = [json.loads(f.read_text()) for f in files]
    speed = [s.get("env_steps_per_s", s.get("steps_per_s")) for s in stats]
    memory = [s["peak_rss_mb_main"] for s in stats if "peak_rss_mb_main" in s]
    return {
        "wall_min": float(np.mean([s["wall_time_s"] for s in stats])) / 60,
        "steps_per_s": float(np.mean(speed)),
        "memory": f"{max(memory):.0f}" if memory else "",
    }


def transfer() -> None:
    lines = []
    for root in sorted(paths.RESULTS_DIR.glob("transfer_*")):
        source, target = root.name.split("_")[1:3]
        files = list(root.glob("eval_*/*/*/metrics.csv"))
        if not files:
            continue
        moved = per_seed(pd.concat([pd.read_csv(f) for f in files]))
        native = _seeds(evaluation_metrics(PREFIX, [target]))
        lines += [f"## Trained on {source}, tested on {target} without retraining\n",
                  "| Demand | Method | n | Transferred, waiting (s) | Retrained, waiting (s) "
                  "| Actuated (s) | Max-pressure (s) | Throughput |", "|" + "---|" * 8]  # fmt: skip
        for profile in PROFILE_ORDER:
            here = native[native.profile == profile].groupby("method").mean_waiting_time.mean()
            for method, m in moved[moved.profile == profile].groupby("method"):
                w = summarize(m.mean_waiting_time)
                lines.append(
                    f"| {profile} | {METHODS[method][0]} | {len(m)} "
                    f"| {w.mean:.1f} ± {w.ci_half_width:.1f} | {here.get(method, np.nan):.1f} "
                    f"| {here.get('actuated', np.nan):.1f} "
                    f"| {here.get('max_pressure', np.nan):.1f} | {m.throughput.mean():.0f} |"
                )
        lines.append("")
    if lines:
        _write("transfer", "\n".join(lines))


def federated() -> None:
    """Sharing nothing, sharing parameters (FedAvg) and sharing raw data (one shared policy)."""
    lines = ["| Scenario | Demand | Scheme | n | Waiting (s) | Δ vs sharing | p (Welch) "
             "| p95 delay (s) "
             "| Throughput | Rounds | MB per round | Total training MB | Raw data leaves? |",
             "|" + "---|" * 13]  # fmt: skip
    panels = []
    for scenario, n in (("corridor", 4), ("grid3x3", 9)):
        root = paths.RESULTS_DIR / f"{PREFIX}_{scenario}"
        rows = []
        for profile in PROFILE_ORDER:
            ref = _scheme(root, profile, "ippo_local")
            for method, (label, _) in FEDERATED.items():
                t = _scheme(root, profile, method)
                if t is None or ref is None:
                    continue
                w = summarize(t.mean_waiting_time)
                d = summarize(t.p95_total_delay)
                is_ref = method == "ippo_local"
                delta = (
                    "" if is_ref else f"{100 * (w.mean / ref.mean_waiting_time.mean() - 1):+.1f} %"
                )
                p = "" if is_ref else f"{welch_p(t.mean_waiting_time, ref.mean_waiting_time):.3f}"
                rounds, per_round, total = _communication(root, method, n)
                raw = "yes" if is_ref else "no"
                lines.append(
                    f"| {scenario} | {profile} | {label} | {len(t)} "
                    f"| {w.mean:.1f} ± {w.ci_half_width:.1f} | {delta} | {p} "
                    f"| {d.mean:.1f} ± {d.ci_half_width:.1f} | {t.throughput.mean():.0f} "
                    f"| {rounds} | {per_round} | {total} | {raw} |"
                )
                rows.append((profile, method, w.mean, w.ci_half_width, total))
        if rows:
            panels.append((scenario, rows))
    if not panels:
        return
    _write("federated", "\n".join(lines))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 3.6),
                             constrained_layout=True, squeeze=False)  # fmt: skip
    for ax, (scenario, rows) in zip(axes[0], panels, strict=True):
        medium = [r for r in rows if r[0] == "medium"][::-1]
        y = np.arange(len(medium))
        ax.errorbar([r[2] for r in medium], y, xerr=[r[3] for r in medium], fmt="none",
                    ecolor=MUTED, elinewidth=1)  # fmt: skip
        ax.scatter([r[2] for r in medium], y, s=60, zorder=3,
                   c=[FEDERATED[r[1]][1] for r in medium], edgecolors="#fcfcfb")  # fmt: skip
        ax.set_yticks(y, [f"{FEDERATED[r[1]][0]}\n{r[4]} MB exchanged" for r in medium])
        ax.set_title(f"{scenario}, medium demand", fontsize=10, loc="left", color=INK)
        ax.set_xlabel("waiting time per vehicle (s), mean and 95 % CI", fontsize=9, color=MUTED)
        _clean(ax)
    _save(fig, "federated")


def _scheme(root: Path, profile: str, method: str) -> pd.DataFrame | None:
    files = list(root.glob(f"eval_{profile}/{method}/*/metrics.csv"))
    if not files:
        return None
    return pd.concat(pd.read_csv(f) for f in files).groupby("seed").mean(numeric_only=True)


def _communication(root: Path, method: str, n_agents: int) -> tuple[str, str, str]:
    """(rounds, MB per round, MB in total) of a training run.

    Parameter sharing moves raw data instead of parameters: every transition of
    every agent (observation, action, reward, done) reaches the central learner.
    """
    if method == "ippo_local":
        files = list((root / method).glob("*/training_stats.json"))
        steps = np.mean([json.loads(f.read_text())["env_steps"] for f in files])
        total = steps * n_agents * (LOCAL_OBS_DIM + 3) * FLOAT_BYTES / 1e6
        return "", "", f"{total:.0f}"
    stats = [json.loads(f.read_text()) for f in (root / method).glob("*/training_stats.json")]
    if method == "independent" or not stats:
        return "0", "0", "0"
    fed = stats[0]["federation"]
    total = (fed["bytes_up"] + fed["bytes_down"]) / 1e6
    return str(fed["rounds"]), f"{fed['bytes_per_round'] / 1e6:.1f}", f"{total:.0f}"


def heatmaps() -> None:
    for file in sorted(paths.RESULTS_DIR.glob("heatmap_*.csv")):
        data = pd.read_csv(file)
        methods = list(dict.fromkeys(data.method))
        top = data.queue.max()
        fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.5),
                                 constrained_layout=True, squeeze=False)  # fmt: skip
        for ax, method in zip(axes[0], methods, strict=True):
            group = data[data.method == method]
            xs, ys = sorted(group.x.unique()), sorted(group.y.unique())
            grid = np.full((len(ys), len(xs)), np.nan)
            for r in group.itertuples():
                grid[ys.index(r.y), xs.index(r.x)] = r.queue
            image = ax.imshow(grid, origin="lower", cmap="Blues", vmin=0, vmax=top)
            ax.set_title(f"{METHODS.get(method, (method,))[0]}\nmean {group.queue.mean():.1f}",
                         fontsize=10, loc="left", color=INK)  # fmt: skip
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
        fig.colorbar(image, ax=axes[0], label="halting vehicles per intersection", shrink=0.8)
        _save(fig, file.stem)


def main() -> None:
    scenarios = _present()
    if scenarios:
        metrics = evaluation_metrics(PREFIX, scenarios)
        metrics = metrics[metrics.method.isin(METHOD_ORDER)]
        (paths.RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
        table = main_tables(metrics)
        for scenario in scenarios:
            metric_figure(table, scenario)
        curves(scenarios)
        _write("training", training_table(PREFIX, scenarios))
        ablation(_seeds(metrics[metrics.profile.isin(PROFILE_ORDER)]))
        robustness(metrics, scenarios)
        scalability(table, scenarios)
    transfer()
    federated()
    heatmaps()


if __name__ == "__main__":
    main()
