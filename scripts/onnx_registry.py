"""
Parikar 2.0 - ONNX Model Registry
Keeps every exported ONNX model in one place (`models_onnx/`) and logs each run,
so retraining never overwrites a previous model.

Layout:
    models_onnx/
        onnx_log.json                      <- machine-readable log of every export
        ONNX_LOG.md                        <- human-readable log (auto-generated)
        <model_key>/
            <model_key>_<run_id>_acc<val>.onnx

`backend/weights/<model_key>.onnx` stays as the "currently served" copy used by the API.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT / "models_onnx"
LOG_JSON = ONNX_DIR / "onnx_log.json"
LOG_MD = ONNX_DIR / "ONNX_LOG.md"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_log() -> list:
    if LOG_JSON.exists():
        try:
            with open(LOG_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _write_markdown(entries: list) -> None:
    lines = [
        "# Parikar 2.0 - ONNX Model Log",
        "",
        "Every exported ONNX model is stored under `models_onnx/<model>/` with a run id.",
        "Retraining adds a new row here - older models are never overwritten.",
        "",
        f"**Total exports logged:** {len(entries)}  ",
        f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| # | Run ID | Model | Val Acc | Train Acc | Latency (GPU) | Size | Epochs | Batch | LR | Train/Val | Train Time | File |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(sorted(entries, key=lambda x: x["run_id"], reverse=True), 1):
        lines.append(
            f"| {i} | `{e['run_id']}` | **{e['display_name']}** | {e['best_val_acc']}% | "
            f"{e.get('final_train_acc', '-')}% | {e.get('latency_ms', '-')} ms | {e['size_mb']} MB | "
            f"{e.get('epochs', '-')} | {e.get('batch_size', '-')} | {e.get('lr', '-')} | "
            f"{e.get('train_samples', '-')}/{e.get('val_samples', '-')} | "
            f"{e.get('training_time_sec', '-')}s | `{e['file']}` |"
        )

    lines += ["", "## SHA-256 Checksums", ""]
    for e in sorted(entries, key=lambda x: x["run_id"], reverse=True):
        lines.append(f"- `{e['file']}` - `{e['sha256']}`")
    lines.append("")

    with open(LOG_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def register_onnx(
    onnx_src: Path,
    model_key: str,
    display_name: str,
    timm_name: str,
    best_val_acc: float,
    final_train_acc: Optional[float] = None,
    latency_ms: Optional[float] = None,
    training_time_sec: Optional[float] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    classes: Optional[list] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Copies an ONNX file into models_onnx/<model_key>/ with a versioned name and logs it."""
    onnx_src = Path(onnx_src)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = ONNX_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    dest = model_dir / f"{model_key}_{run_id}_acc{best_val_acc:.2f}.onnx"
    shutil.copy2(onnx_src, dest)

    entry = {
        "run_id": run_id,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
        "display_name": display_name,
        "timm_name": timm_name,
        "best_val_acc": best_val_acc,
        "final_train_acc": final_train_acc,
        "latency_ms": latency_ms,
        "training_time_sec": training_time_sec,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "classes": classes,
        "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
        "sha256": _sha256(dest),
        "file": dest.relative_to(ONNX_DIR).as_posix(),
    }

    entries = [e for e in _load_log() if not (e["model_key"] == model_key and e["run_id"] == run_id)]
    entries.append(entry)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    _write_markdown(entries)

    print(f"Registered ONNX -> {dest} (log: {LOG_MD})", flush=True)
    return dest
