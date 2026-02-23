# cleanup_artifacts.ps1
# Run this once to clean up .dsl.txt, .analysis.json, and chained .dsl.dsl... files
# from the clipboard-exports folder left behind by previous pipeline runs.

$inbox = "C:\BlueprintLLM\raw-data\clipboard-exports"

Write-Host "Scanning $inbox for artifact files..." -ForegroundColor Yellow

$artifacts = Get-ChildItem $inbox -File | Where-Object {
    $_.Name -match "\.dsl" -or
    $_.Name -match "\.analysis" -or
    $_.Name -match "\.dsl\.txt$"
}

if ($artifacts.Count -eq 0) {
    Write-Host "No artifact files found. Folder is clean." -ForegroundColor Green
    exit
}

Write-Host "Found $($artifacts.Count) artifact file(s) to remove:" -ForegroundColor Yellow
$artifacts | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }

$confirm = Read-Host "`nDelete these files? (y/n)"
if ($confirm -eq "y") {
    $artifacts | Remove-Item -Force
    Write-Host "Deleted $($artifacts.Count) files." -ForegroundColor Green
} else {
    Write-Host "Cancelled." -ForegroundColor Yellow
}
