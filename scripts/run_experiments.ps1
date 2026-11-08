# Launch a batch of cross-validation runs, one PowerShell window each.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1            # batch 1
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Batch 2
#
# Each window shows its run's progress live and stays open when the run ends.
# Logs are also written to runs\logs\<date>_<name>.log and reports to
# runs\<date>_<name>\report.md; runs\README.md compares every run.
# The windows do not depend on VS Code: closing the editor does not stop them.

param([int]$Batch = 1, [int]$Threads = 4)

$root = Split-Path -Parent $PSScriptRoot
$common = "--seeds 0 1 --plots summary --threads $Threads"

$batches = @{
    # Batch 1: rig clips only (--external off), one change at a time.
    1 = [ordered]@{
        "baseline"        = "--external off"
        "margin"          = "--external off --ignore-partial"
        "margin-aug3"     = "--external off --ignore-partial --augment 3"
        "margin-synth200" = "--external off --ignore-partial --synthetic 200"
        "margin-tcn"      = "--external off --ignore-partial --arch tcn"
    }
    # Batch 2: does CAUCAFall help? Every run is scored on the rig clips only;
    # CAUCAFall clips are added to training (or, in the last run, tested too).
    # Needs data/output/caucafall/*.json (see docs/external-data.md).
    2 = [ordered]@{
        "rig-only"            = "--external off --ignore-partial"
        "plus-cauca"          = "--external train --ignore-partial"
        "plus-cauca-aug3"     = "--external train --ignore-partial --augment 3"
        "rig-only-win18"      = "--external off --ignore-partial --window-size 18"
        "cauca-in-folds"      = "--external cv --ignore-partial"
    }
}

if (-not $batches.ContainsKey($Batch)) {
    Write-Host "No batch $Batch defined in $PSCommandPath" -ForegroundColor Red
    exit 1
}

foreach ($name in $batches[$Batch].Keys) {
    $options = $batches[$Batch][$name]
    $command = "`$Host.UI.RawUI.WindowTitle = 'train: $name'; " +
               "Set-Location '$root'; `$env:PYTHONUNBUFFERED = '1'; " +
               "uv run scripts/train_cv.py --name $name $options $common; " +
               "Write-Host ''; Write-Host 'Run $name finished. This window can be closed.' -ForegroundColor Green"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $command
    Write-Host "Started $name ($options)"
}
Write-Host "Logs: $root\runs\logs   Reports: $root\runs\<date>_<name>\report.md"
