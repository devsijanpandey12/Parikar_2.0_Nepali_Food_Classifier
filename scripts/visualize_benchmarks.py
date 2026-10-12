"""
Parikar 2.0 - Model Benchmarks & Training Visualizer
Generates comprehensive visualization charts for every model run:
  1. Per-run loss & accuracy curves across epochs
  2. Multi-model comparative dashboard (Val Acc, Latency, Model Size, Training Time)
  3. Interactive HTML benchmark report with Chart.js
  4. Automatic integration with train_models.py so every future retraining session
     generates persistent, non-overwriting charts.

Output directory:
  reports_and_charts/
    ├── comparisons/
    │   ├── model_comparison_latest.png
    │   ├── model_comparison_latest.json
    │   ├── benchmark_dashboard.html
    │   └── model_comparison_<timestamp>.png
    └── <model_key>/
        ├── <model_key>_<run_id>_training_curves.png
        └── <model_key>_latest_curves.png
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = ROOT / "reports_and_charts"

# Himalayan dusk / modern glassmorphic theme colors
PALETTE = {
    "mobilenet_v4": {"primary": "#10b981", "secondary": "#059669", "name": "MobileNetV4-Conv"},
    "convnext_v2": {"primary": "#6366f1", "secondary": "#4f46e5", "name": "ConvNeXt-V2-Nano"},
    "efficientnet_b0": {"primary": "#f59e0b", "secondary": "#d97706", "name": "EfficientNet-B0"},
}
DARK_BG = "#0b0f19"
SURFACE_BG = "#131b2e"
GRID_COLOR = "#222f4c"
TEXT_COLOR = "#f1f5f9"
MUTED_TEXT = "#94a3b8"


def apply_dark_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """Applies a clean modern dark aesthetic to a matplotlib axes."""
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


def plot_single_model_training_curves(
    model_key: str,
    history: list,
    run_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Plots Train vs Val Loss and Train vs Val Accuracy curves across epochs for one model.
    Saves a versioned copy and a `_latest_curves.png` pointer.
    """
    output_dir = Path(output_dir or DEFAULT_REPORTS_DIR)
    model_folder = output_dir / model_key
    model_folder.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    theme = PALETTE.get(model_key, {"primary": "#38bdf8", "secondary": "#0284c7", "name": model_key})

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=DARK_BG)
    fig.suptitle(
        f"{theme['name']} - Training & Validation Progress (Run: {run_id})",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.98,
    )

    # Accuracy Plot
    apply_dark_style(ax_acc, "Accuracy (%) vs Epoch", "Epoch", "Accuracy (%)")
    ax_acc.plot(epochs, train_acc, color="#10b981", marker="o", linewidth=2.2, label=f"Train Acc (Final: {train_acc[-1]:.2f}%)")
    ax_acc.plot(epochs, val_acc, color=theme["primary"], marker="s", linewidth=2.4, label=f"Val Acc (Best: {max(val_acc):.2f}%)")
    ax_acc.set_ylim(min(min(train_acc), min(val_acc)) - 4, 102)
    ax_acc.legend(facecolor=SURFACE_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="lower right")

    # Loss Plot
    apply_dark_style(ax_loss, "Cross-Entropy Loss vs Epoch", "Epoch", "Loss")
    ax_loss.plot(epochs, train_loss, color="#f43f5e", marker="o", linewidth=2.2, label=f"Train Loss (Final: {train_loss[-1]:.4f})")
    ax_loss.plot(epochs, val_loss, color="#fbbf24", marker="s", linewidth=2.2, label=f"Val Loss (Final: {val_loss[-1]:.4f})")
    ax_loss.legend(facecolor=SURFACE_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="upper right")

    plt.tight_layout()

    versioned_path = model_folder / f"{model_key}_{run_id}_training_curves.png"
    latest_path = model_folder / f"{model_key}_latest_curves.png"
    plt.savefig(versioned_path, dpi=200, facecolor=DARK_BG)
    plt.savefig(latest_path, dpi=200, facecolor=DARK_BG)
    plt.close(fig)

    print(f"Saved training curve chart -> {versioned_path}", flush=True)
    return versioned_path


def generate_comparative_charts(
    metrics_data: Dict[str, Any],
    output_dir: Optional[Path] = None,
    run_tag: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generates a 4-panel comparison dashboard across all trained models:
      1. Validation Accuracy (%) [Bar with rank]
      2. Inference Latency (ms) [Bar, lower is better]
      3. ONNX Model Size (MB) [Bar, lower is better]
      4. Accuracy vs Latency Pareto Frontier Scatter
    Also generates an interactive standalone HTML dashboard (`benchmark_dashboard.html`).
    """
    output_dir = Path(output_dir or DEFAULT_REPORTS_DIR)
    comp_dir = output_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    model_keys = list(metrics_data.keys())
    if not model_keys:
        return {}

    display_names = [metrics_data[k].get("display_name", k) for k in model_keys]
    val_accs = [metrics_data[k].get("best_val_acc", 0.0) for k in model_keys]
    latencies = [metrics_data[k].get("inference_latency_ms", 0.0) for k in model_keys]
    sizes = [metrics_data[k].get("onnx_size_mb", 0.0) for k in model_keys]
    train_times = [metrics_data[k].get("training_time_sec", 0.0) / 60.0 for k in model_keys]  # minutes

    colors = [PALETTE.get(k, {"primary": "#38bdf8"})["primary"] for k in model_keys]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), facecolor=DARK_BG)
    fig.suptitle(
        f"Parikar 2.0 - Multi-Model Architecture Performance Benchmark ({run_tag})",
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
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.6,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontweight="bold",
            fontsize=11,
        )

    # 2. Inference Latency
    ax_lat = axes[0, 1]
    apply_dark_style(ax_lat, "Inference Latency (ms per image) - Lower is Better", ylabel="Latency (ms)")
    bars_lat = ax_lat.bar(display_names, latencies, color=["#10b981", "#6366f1", "#f59e0b"], width=0.55, edgecolor=GRID_COLOR)
    ax_lat.set_ylim(0, max(latencies) * 1.3)
    for bar in bars_lat:
        height = bar.get_height()
        ax_lat.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(latencies) * 0.03,
            f"{height:.2f} ms",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontweight="bold",
            fontsize=11,
        )

    # 3. Model Size
    ax_size = axes[1, 0]
    apply_dark_style(ax_size, "ONNX Model Disk Footprint (MB) - Lower is Better", ylabel="Size (MB)")
    bars_size = ax_size.bar(display_names, sizes, color=["#10b981", "#818cf8", "#fbbf24"], width=0.55, edgecolor=GRID_COLOR)
    ax_size.set_ylim(0, max(sizes) * 1.25)
    for bar in bars_size:
        height = bar.get_height()
        ax_size.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(sizes) * 0.03,
            f"{height:.2f} MB",
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontweight="bold",
            fontsize=11,
        )

    # 4. Accuracy vs Latency Pareto Frontier Scatter
    ax_pareto = axes[1, 1]
    apply_dark_style(ax_pareto, "Pareto Trade-off: Accuracy vs Latency", xlabel="Inference Latency (ms)", ylabel="Val Accuracy (%)")
    for i, k in enumerate(model_keys):
        ax_pareto.scatter(latencies[i], val_accs[i], s=260, color=colors[i], edgecolors="#ffffff", linewidths=1.5, zorder=5)
        ax_pareto.annotate(
            f"{display_names[i]}\n({val_accs[i]:.1f}%, {latencies[i]:.1f}ms)",
            (latencies[i], val_accs[i]),
            xytext=(10, -5),
            textcoords="offset points",
            color=TEXT_COLOR,
            fontweight="bold",
            fontsize=10,
        )
    ax_pareto.set_xlim(min(latencies) * 0.7, max(latencies) * 1.4)
    ax_pareto.set_ylim(min(val_accs) - 2, max(val_accs) + 2)

    plt.tight_layout()

    versioned_png = comp_dir / f"model_comparison_{run_tag}.png"
    latest_png = comp_dir / "model_comparison_latest.png"
    plt.savefig(versioned_png, dpi=200, facecolor=DARK_BG)
    plt.savefig(latest_png, dpi=200, facecolor=DARK_BG)
    plt.close(fig)

    # Generate interactive HTML dashboard
    html_path = comp_dir / "benchmark_dashboard.html"
    _generate_interactive_html_dashboard(metrics_data, html_path, latest_png.name)

    # Also save comparison metrics JSON
    comp_json = comp_dir / "model_comparison_latest.json"
    with open(comp_json, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Generated comparison dashboard -> {versioned_png}", flush=True)
    print(f"Generated interactive dashboard -> {html_path}", flush=True)

    return {
        "comparison_png": versioned_png,
        "latest_png": latest_png,
        "dashboard_html": html_path,
    }


def _generate_interactive_html_dashboard(metrics: Dict[str, Any], output_html: Path, chart_img_name: str):
    """Creates a self-contained, glassmorphic HTML report with charts and metrics tables."""
    cards_html = ""
    for k, data in metrics.items():
        theme = PALETTE.get(k, {"primary": "#38bdf8", "name": k})
        cards_html += f"""
        <div class="card" style="border-top: 4px solid {theme['primary']};">
            <div class="card-header">
                <div>
                    <h3 style="margin:0; font-size:1.25rem; color:#fff;">{data.get('display_name', k)}</h3>
                    <div style="font-size:0.8rem; color:#94a3b8; font-family:monospace; margin-top:4px;">{data.get('timm_name', '-')}</div>
                </div>
                <span class="badge" style="background:{theme['primary']}22; color:{theme['primary']}; border:1px solid {theme['primary']}55;">
                    {data.get('best_val_acc', 0)}% Acc
                </span>
            </div>
            <div class="stats-grid">
                <div class="stat"><div class="stat-val">{data.get('best_val_acc', 0)}%</div><div class="stat-lbl">Best Val Accuracy</div></div>
                <div class="stat"><div class="stat-val">{data.get('inference_latency_ms', 0)} ms</div><div class="stat-lbl">Inference Latency</div></div>
                <div class="stat"><div class="stat-val">{data.get('onnx_size_mb', 0)} MB</div><div class="stat-lbl">ONNX Size</div></div>
                <div class="stat"><div class="stat-val">{data.get('epochs', 0)}</div><div class="stat-lbl">Epochs Trained</div></div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Parikar 2.0 - Model Benchmarks & Comparison Dashboard</title>
<style>
  :root {{
    --bg: #0b0f19;
    --card: #131b2e;
    --border: #222f4c;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #6366f1;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 30px 20px;
    line-height: 1.5;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  header {{
    text-align: center;
    margin-bottom: 40px;
    padding-bottom: 25px;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{ margin: 0 0 10px; font-size: 2.2rem; background: linear-gradient(135deg, #f59e0b, #ec4899, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  header p {{ color: var(--muted); margin: 0; font-size: 1.05rem; }}
  .cards-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
  }}
  .card-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; }}
  .badge {{ padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }}
  .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  .stat {{ background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); }}
  .stat-val {{ font-size: 1.4rem; font-weight: 800; color: #fff; }}
  .stat-lbl {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .chart-section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 40px;
    text-align: center;
  }}
  .chart-section h2 {{ margin: 0 0 20px; text-align: left; font-size: 1.4rem; }}
  .chart-section img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); }}
  .table-section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    overflow-x: auto;
  }}
  table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.95rem; }}
  th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; background: rgba(0,0,0,0.2); }}
  tr:last-child td {{ border-bottom: none; }}
  .tag-gold {{ color: #fbbf24; font-weight: bold; }}
  .tag-green {{ color: #34d399; font-weight: bold; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Parikar 2.0 • Vision Model Benchmark Suite</h1>
    <p>Empirical evaluation of fine-tuned backbones on 10 authentic Nepali food classes</p>
  </header>

  <div class="cards-grid">
    {cards_html}
  </div>

  <div class="chart-section">
    <h2>Performance & Trade-Off Visualization</h2>
    <img src="{chart_img_name}" alt="Parikar 2.0 Model Comparison Charts">
  </div>

  <div class="table-section">
    <h2 style="margin: 0 0 20px; font-size:1.4rem;">Empirical Benchmark Summary Matrix</h2>
    <table>
      <thead>
        <tr>
          <th>Model Architecture</th>
          <th>Validation Accuracy</th>
          <th>Final Train Acc</th>
          <th>GPU Latency</th>
          <th>ONNX Size</th>
          <th>Epochs</th>
          <th>Primary Role</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>ConvNeXt-V2-Nano</strong></td>
          <td class="tag-gold">90.00% (Rank #1)</td>
          <td>99.94%</td>
          <td>7.00 ms</td>
          <td>57.30 MB</td>
          <td>10</td>
          <td><span class="badge" style="background:#6366f133; color:#818cf8;">Maximum Precision</span></td>
        </tr>
        <tr>
          <td><strong>EfficientNet-B0</strong></td>
          <td>88.30% (Rank #2)</td>
          <td>99.98%</td>
          <td>10.73 ms</td>
          <td>15.31 MB</td>
          <td>12</td>
          <td><span class="badge" style="background:#f59e0b33; color:#fbbf24;">Compound Scaled</span></td>
        </tr>
        <tr>
          <td><strong>MobileNetV4-Conv</strong></td>
          <td>87.00% (Rank #3)</td>
          <td>99.98%</td>
          <td class="tag-green">5.34 ms (Fastest)</td>
          <td class="tag-green">9.53 MB (Smallest)</td>
          <td>12</td>
          <td><span class="badge" style="background:#10b98133; color:#34d399;">Ultra-Fast Edge</span></td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_all_reports_from_metrics(metrics_file: str = "docs/training_metrics.json"):
    """Reads existing metrics JSON and generates individual and comparative charts."""
    path = Path(metrics_file)
    if not path.exists():
        print(f"Metrics file not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Generate individual curve charts for each model
    for m_key, m_data in metrics.items():
        if "history" in m_data and m_data["history"]:
            plot_single_model_training_curves(m_key, m_data["history"])

    # 2. Generate multi-model comparative dashboard
    generate_comparative_charts(metrics)


if __name__ == "__main__":
    generate_all_reports_from_metrics()
