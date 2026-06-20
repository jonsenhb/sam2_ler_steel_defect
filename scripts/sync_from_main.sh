#!/usr/bin/env bash
# Sync code and results from main research repo into the GitHub release bundle.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST="$ROOT/release/sam2_ler_github"

echo "Syncing from $ROOT -> $DEST"

copy() { mkdir -p "$(dirname "$2")"; cp "$1" "$2"; }

# Core modules
copy "$ROOT/train_multiclass.py" "$DEST/code/sam2_ler/train_multiclass.py"
copy "$ROOT/dataset.py" "$DEST/code/sam2_ler/dataset.py"
copy "$ROOT/train.py" "$DEST/code/sam2_ler/train.py"
copy "$ROOT/segmentation_metrics.py" "$DEST/code/sam2_ler/segmentation_metrics.py"
copy "$ROOT/asi_metrics.py" "$DEST/code/analysis/asi_metrics.py"
copy "$ROOT/asi_per_image.py" "$DEST/code/analysis/asi_per_image.py"
copy "$ROOT/thesis_validation.py" "$DEST/code/analysis/thesis_validation.py"
copy "$ROOT/run_asi_experiments.py" "$DEST/code/analysis/run_asi_experiments.py"

# Pipelines
for f in paper_de_pipeline.py paper_de_report.py paper_sensors_figures.py; do
  copy "$ROOT/$f" "$DEST/code/pipelines/$f"
done

# Probes
for f in probe_cb_peft.py probe_dsa_fpn.py probe_de_common.py probe_rad_loss.py; do
  [ -f "$ROOT/$f" ] && copy "$ROOT/$f" "$DEST/code/probes/$f"
done

# Results JSON (no checkpoints)
rsync -a --delete --include='*/' --include='*.json' --include='*.csv' --include='*.md' --include='*.tex' \
  --exclude='*' "$ROOT/outputs/paper_de/" "$DEST/results/paper_de/" 2>/dev/null || true
rsync -a --include='*/' --include='*.json' --exclude='*' \
  "$ROOT/outputs/thesis_validation/" "$DEST/results/thesis_validation/" 2>/dev/null || true

# Regenerate manifest
python3 - << 'PY'
import hashlib, json
from pathlib import Path
dest = Path("release/sam2_ler_github/results")
files = sorted(dest.rglob("*.json"))
manifest = []
for p in files:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest.append({"path": str(p.relative_to(dest.parent)), "sha256": h})
Path("release/sam2_ler_github/results/MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"MANIFEST: {len(manifest)} files")
PY

echo "Sync complete."
