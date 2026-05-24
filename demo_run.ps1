# Run the 5-step terminal demo end-to-end.
# Usage: .\demo_run.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n=== STEP 1 — Masking visualisation ===" -ForegroundColor Cyan
python scripts/demo_masking.py --n 4 --out assets/masking_demo.png
Invoke-Item assets/masking_demo.png

Write-Host "`n=== STEP 2 — Loss curve ===" -ForegroundColor Cyan
python scripts/plot_loss.py --log checkpoints/train_log.csv --out assets/loss_curve.png
Invoke-Item assets/loss_curve.png

Write-Host "`n=== STEP 3 — Training log (first 5 + last 5 steps) ===" -ForegroundColor Cyan
Get-Content checkpoints/train_log.csv -Head 5
Write-Host "..."
Get-Content checkpoints/train_log.csv -Tail 5

Write-Host "`n=== STEP 4 — Linear probing ===" -ForegroundColor Cyan
python -m src.evaluate --checkpoint checkpoints/ijepa_smoke.pt --epochs 20

Write-Host "`n=== STEP 5 — t-SNE plot ===" -ForegroundColor Cyan
Invoke-Item assets/tsne.png

Write-Host "`n=== DONE ===" -ForegroundColor Green
