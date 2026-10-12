"""
Parikar 2.0 - Model Training Run Ledger & Visualizer
Maintains a chronological session ledger of every training session:
  - Session 1 (Initial Base Training - 10,000 images, 3 models)
  - Session 2 (Retraining / Fine-Tuning run...)
  - Session 3 ...

Outputs:
  reports_and_charts/
    ├── TRAINING_RUNS_LOG.md           <- Complete chronological log table with links to charts
    ├── training_runs_history.json     <- Machine-readable history across all training sessions
    ├── session_1/                     <- Specific session folder
    │   ├── model_comparison.png
    │   ├── benchmark_dashboard.html
    │   ├── mobilenet_v4_curves.png
    │   ├── convnext_v2_curves.png
    │   └── efficientnet_b0_curves.png
    ├── session_2/                     <- Future retraining sessions automatically created!
    └── comparisons/
        ├── model_comparison_latest.png
        └── benchmark_dashboard.html
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports_and_charts"
RUNS_JSON = REPORTS_DIR / "training_runs_history.json"
RUNS_MD = REPORTS_DIR / "TRAINING_RUNS_LOG.md"

PALETTE = {
    "mobilenet_v4": {"primary": "#10b981", "name": "MobileNetV4-Conv"},
    "convnext_v2": {"primary": "#6366f1", "name": "ConvNeXt-V2-Nano"},
    "efficientnet_b0": {"primary": "#f59e0b", "name": "EfficientNet-B0"},
}
DARK_BG = "#0b0f19"
SURFACE_BG = "#131b2e"
GRID_COLOR = "#222f4c"
TEXT_COLOR = "#f1f5f9"
MUTED_TEXT = "#94a3b8"


def _load_history() -> List[Dict[str, Any]]:
    if RUNS_JSON.exists():
        try:
            with open(RUNS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(history: List[Dict[str, Any]]):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _write_markdown_ledger(history: List[Dict[str, Any]]):
    lines = [
        "# Parikar 2.0 • Chronological Training & Retraining Ledger 📈",
        "",
        "This ledger records every training and retraining session in sequential order (`Session 1`, `Session 2`, etc.).",
        "Every session stores its own independent performance charts and curves without overwriting previous sessions.",
        "",
        f"**Total Training Sessions Logged:** {len(history)}  ",
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "---",
        "",
    ]

    for session in history:
        s_num = session["session_number"]
        s_title = session.get("title", f"Training Session #{s_num}")
        s_date = session.get("timestamp", "-")
        s_note = session.get("notes", "Model training run")
        s_dir = session.get("folder_name", f"session_{s_num}")

        lines += [
            f"## 🏆 Session {s_num}: {s_title}",
            f"- **Date & Time:** {s_date}",
            f"- **Session Directory:** [`reports_and_charts/{s_dir}/`](./{s_dir}/)",
            f"- **Description / Notes:** {s_note}",
            "",
            "| Model Architecture | Best Val Acc | Final Train Acc | GPU Latency | ONNX Size | Epochs | Training Time | Curves Chart |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for m_key, m in session.get("models", {}).items():
            curves_link = f"[`{m_key}_curves.png`](./{s_dir}/{m_key}_curves.png)" if (REPORTS_DIR / s_dir / f"{m_key}_curves.png").exists() else "-"
            lines.append(
                f"| **{m.get('display_name', m_key)}** | **{m.get('best_val_acc', '-')}%** | "
                f"{m.get('final_train_acc', '-')}% | {m.get('inference_latency_ms', '-')} ms | "
                f"{m.get('onnx_size_mb', '-')} MB | {m.get('epochs', '-')} | "
                f"{round(m.get('training_time_sec', 0)/60, 1) if m.get('training_time_sec') else '-'} min | {curves_link} |"
            )

        lines += [
            "",
            f"**Session Visualizations:**",
            f"- 📊 [Comparative Benchmark Dashboard Chart](./{s_dir}/model_comparison.png)",
            f"- 🌐 [Interactive HTML Dashboard](./{s_dir}/benchmark_dashboard.html)",
            "",
            "---",
            "",
        ]

    with open(RUNS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def apply_dark_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold", color=MUTED_TEXT, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="semibold", color=MUTED_TEXT, labelpad=8)


def plot_session_model_curves(model_key: str, history: list, session_dir: Path, session_num: int):
    """Plots and saves loss/accuracy curves for a model inside the session folder."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    theme = PALETTE.get(model_key, {"primary": "#38bdf8", "name": model_key})

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK_BG)
    fig.suptitle(
        f"Session {session_num} • {theme['name']} - Training & Validation Progress",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.98,
    )

    apply_dark_style(ax_acc, "Accuracy (%) vs Epoch", "Epoch", "Accuracy (%)")
    ax_acc.plot(epochs, train_acc, color="#10b981", marker="o", linewidth=2.2, label=f"Train Acc (Final: {train_acc[-1]:.2f}%)")
    ax_acc.plot(epochs, val_acc, color=theme["primary"], marker="s", linewidth=2.4, label=f"Val Acc (Best: {max(val_acc):.2f}%)")
    ax_acc.set_ylim(min(min(train_acc), min(val_acc)) - 4, 102)
    ax_acc.legend(facecolor=SURFACE_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="lower right")

    apply_dark_style(ax_loss, "Loss vs Epoch", "Epoch", "Loss")
    ax_loss.plot(epochs, train_loss, color="#f43f5e", marker="o", linewidth=2.2, label=f"Train Loss (Final: {train_loss[-1]:.4f})")
    ax_loss.plot(epochs, val_loss, color="#fbbf24", marker="s", linewidth=2.2, label=f"Val Loss (Final: {val_loss[-1]:.4f})")
    ax_loss.legend(facecolor=SURFACE_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="upper right")

    plt.tight_layout()
    out_path = session_dir / f"{model_key}_curves.png"
    plt.savefig(out_path, dpi=200, facecolor=DARK_BG)
    plt.close(fig)


def plot_session_comparison(metrics_data: Dict[str, Any], session_dir: Path, session_num: int, title_suffix: str = ""):
    """Generates the 4-panel comparison dashboard for this session."""
    model_keys = list(metrics_data.keys())
    if not model_keys:
        return

    display_names = [metrics_data[k].get("display_name", k) for k in model_keys]
    val_accs = [metrics_data[k].get("best_val_acc", 0.0) for k in model_keys]
    latencies = [metrics_data[k].get("inference_latency_ms", 0.0) for k in model_keys]
    sizes = [metrics_data[k].get("onnx_size_mb", 0.0) for k in model_keys]
    colors = [PALETTE.get(k, {"primary": "#38bdf8"})["primary"] for k in model_keys]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), facecolor=DARK_BG)
    fig.suptitle(
        f"Parikar 2.0 • Session {session_num} Model Architecture Benchmark {title_suffix}",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.98,
    )

    # 1. Validation Accuracy
    ax_acc = axes[0, 0]
    apply_dark_style(ax_acc, "Top-1 Validation Accuracy (%) - Higher is Better", ylabel="Validation Accuracy (%)")
    bars_acc = ax_acc.bar(display_names, val_accs, color=colors, width=0.55, edgecolor=GRID_COLOR)
    ax_acc.set_ylim(80, 100)
    for bar in bars_acc:
        height = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f"{height:.2f}%", ha="center", va="bottom", color=TEXT_COLOR, fontweight="bold", fontsize=11)

    # 2. Latency
    ax_lat = axes[0, 1]
    apply_dark_style(ax_lat, "Inference Latency (ms per image) - Lower is Better", ylabel="Latency (ms)")
    bars_lat = ax_lat.bar(display_names, latencies, color=["#10b981", "#6366f1", "#f59e0b"], width=0.55, edgecolor=GRID_COLOR)
    ax_lat.set_ylim(0, max(latencies) * 1.3)
    for bar in bars_lat:
        height = bar.get_height()
        ax_lat.text(bar.get_x() + bar.get_width()/2.0, height + max(latencies)*0.03, f"{height:.2f} ms", ha="center", va="bottom", color=TEXT_COLOR, fontweight="bold", fontsize=11)

    # 3. Model Size
    ax_size = axes[1, 0]
    apply_dark_style(ax_size, "ONNX Model Disk Footprint (MB) - Lower is Better", ylabel="Size (MB)")
    bars_size = ax_size.bar(display_names, sizes, color=["#10b981", "#818cf8", "#fbbf24"], width=0.55, edgecolor=GRID_COLOR)
    ax_size.set_ylim(0, max(sizes) * 1.25)
    for bar in bars_size:
        height = bar.get_height()
        ax_size.text(bar.get_x() + bar.get_width()/2.0, height + max(sizes)*0.03, f"{height:.2f} MB", ha="center", va="bottom", color=TEXT_COLOR, fontweight="bold", fontsize=11)

    # 4. Pareto Frontier
    ax_pareto = axes[1, 1]
    apply_dark_style(ax_pareto, "Pareto Trade-off: Accuracy vs Latency", xlabel="Inference Latency (ms)", ylabel="Val Accuracy (%)")
    for i, k in enumerate(model_keys):
        ax_pareto.scatter(latencies[i], val_accs[i], s=260, color=colors[i], edgecolors="#ffffff", linewidths=1.5, zorder=5)
        ax_pareto.annotate(f"{display_names[i]}\n({val_accs[i]:.1f}%, {latencies[i]:.1f}ms)", (latencies[i], val_accs[i]), xytext=(10, -5), textcoords="offset points", color=TEXT_COLOR, fontweight="bold", fontsize=10)
    ax_pareto.set_xlim(min(latencies) * 0.7, max(latencies) * 1.4)
    ax_pareto.set_ylim(min(val_accs) - 2, max(val_accs) + 2)

    plt.tight_layout()
    out_png = session_dir / "model_comparison.png"
    plt.savefig(out_png, dpi=200, facecolor=DARK_BG)
    plt.close(fig)

    # Also update latest copy in reports_and_charts/comparisons
    comp_dir = REPORTS_DIR / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_png, comp_dir / "model_comparison_latest.png")


def record_training_session(
    metrics_data: Dict[str, Any],
    session_title: Optional[str] = None,
    session_notes: Optional[str] = None,
) -> int:
    """
    Records a training session chronologically (Session 1, Session 2, ...),
    creates a dedicated session directory `reports_and_charts/session_N/`,
    generates all session charts, and updates the markdown ledger.
    """
    history = _load_history()
    session_num = len(history) + 1
    session_folder_name = f"session_{session_num}"
    session_dir = REPORTS_DIR / session_folder_name
    session_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = session_title or (f"1st Initial Training Run (10k Dataset)" if session_num == 1 else f"Retraining Run #{session_num}")
    notes = session_notes or (
        "Initial complete training across all 10 authentic Nepali food classes with 10k dataset."
        if session_num == 1 else f"Subsequent retraining session #{session_num}."
    )

    # 1. Plot individual curves inside session folder
    for m_key, m_val in metrics_data.items():
        if "history" in m_val:
            plot_session_model_curves(m_key, m_val["history"], session_dir, session_num)

    # 2. Plot comparison dashboard inside session folder
    plot_session_comparison(metrics_data, session_dir, session_num)

    # 3. Create interactive HTML report for this session
    try:
        from visualize_benchmarks import _generate_interactive_html_dashboard
    except ImportError:
        from scripts.visualize_benchmarks import _generate_interactive_html_dashboard
    _generate_interactive_html_dashboard(metrics_data, session_dir / "benchmark_dashboard.html", "model_comparison.png")
    shutil.copy2(session_dir / "benchmark_dashboard.html", REPORTS_DIR / "comparisons" / "benchmark_dashboard.html")

    # 4. Save session JSON
    with open(session_dir / "session_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 5. Append to history ledger
    entry = {
        "session_number": session_num,
        "folder_name": session_folder_name,
        "title": title,
        "notes": notes,
        "timestamp": timestamp,
        "models": {
            k: {
                "display_name": v.get("display_name", k),
                "timm_name": v.get("timm_name", "-"),
                "best_val_acc": v.get("best_val_acc"),
                "final_train_acc": v.get("final_train_acc"),
                "inference_latency_ms": v.get("inference_latency_ms"),
                "onnx_size_mb": v.get("onnx_size_mb"),
                "epochs": v.get("epochs"),
                "training_time_sec": v.get("training_time_sec"),
            }
            for k, v in metrics_data.items()
        }
    }
    history.append(entry)
    _save_history(history)
    _write_markdown_ledger(history)

    print(f"\n==================================================", flush=True)
    print(f"Logged Training Session #{session_num}: '{title}'", flush=True)
    print(f"Saved to: {session_dir}", flush=True)
    print(f"Ledger updated: {RUNS_MD}", flush=True)
    print(f"==================================================\n", flush=True)
    return session_num


def init_or_update():
    """Initializes Session 1 from existing training metrics if history is empty."""
    metrics_file = ROOT / "docs" / "training_metrics.json"
    if not metrics_file.exists():
        print("No training metrics found.")
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    history = _load_history()
    if not history:
        print("Initializing Session 1 from base training run...")
        record_training_session(
            metrics_data=metrics,
            session_title="1st Training Run • Base 10,000 Dataset",
            session_notes="Initial baseline training across MobileNetV4, ConvNeXt-V2, and EfficientNet-B0 on 10,000 balanced Nepali food images."
        )
    else:
        print(f"Existing history has {len(history)} sessions.")


if __name__ == "__main__":
    init_or_update()
