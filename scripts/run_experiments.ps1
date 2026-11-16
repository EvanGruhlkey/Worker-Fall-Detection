# Launch a batch of cross-validation runs, one PowerShell window each.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1            # batch 1
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Batch 2
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Batch 3 -Parallel 3   # Trial 2
#
# Each window shows its run's progress live and stays open when the run ends.
# Logs are also written to runs\logs\<date>_<name>.log and reports to
# runs\<date>_<name>\report.md; runs\README.md compares every run.
# The windows do not depend on VS Code: closing the editor does not stop them.

param([int]$Batch = 1, [int]$Threads = 4, [int]$Parallel = 0)

$root = Split-Path -Parent $PSScriptRoot
# Trial 1 (batches 1-2) used 2 seeds; Trial 2 (batch 3+) uses 3, since one fall
# caught or missed moves the rate several points and 3 seeds show that noise.
$seeds = if ($Batch -ge 3) { "0 1 2" } else { "0 1" }
$common = "--seeds $seeds --plots summary --threads $Threads"

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
    # Batch 3 (Trial 2): the new baseline is Trial 1's best setup on all the
    # new data (night rig clips, team gym falls, CAUCAFall without its 12 slow
    # falls). Every other run changes exactly one thing from it. Compare runs on
    # the "Honest" and "On Trial 1 clips" columns of runs\README.md.
    3 = [ordered]@{
        "t2-base"          = "--external train --ignore-partial"
        "t2-no-gym"        = "--external train --ignore-partial --drop-source gym"
        "t2-with-slow"     = "--external train --ignore-partial --include-test-only"
        "t2-win18"         = "--external train --ignore-partial --window-size 18"
        "t2-drop015"       = "--external train --ignore-partial --min-drop 0.15"
        "t2-smooth01"      = "--external train --ignore-partial --label-smoothing 0.1"
        "t2-low-x3"        = "--external train --ignore-partial --hard-negative-weight 3"
        "t2-gap12"         = "--external train --ignore-partial --max-gap 12"
        "t2-dedupe"        = "--external train --ignore-partial --merge-duplicates"
        "t2-mirror"        = "--external train --ignore-partial --augment-mode mirror"
    }
    # Batch 4 (Trial 2): the batch 3 changes that helped, combined: mirrored
    # training copies, the fast-drop check and duplicate-box removal.
    4 = [ordered]@{
        "t2-best"          = "--external train --ignore-partial --augment-mode mirror --min-drop 0.15 --merge-duplicates"
    }
}

if (-not $batches.ContainsKey($Batch)) {
    Write-Host "No batch $Batch defined in $PSCommandPath" -ForegroundColor Red
    exit 1
}

if ($batches[$Batch].Count -eq 0) {
    Write-Host "Batch $Batch has no runs yet; add them in $PSCommandPath" -ForegroundColor Yellow
    exit 1
}

# -Parallel N keeps at most N runs training at once (0 = start all at once, as
# before); the rest start as earlier ones finish. Ten at once fight for the CPU.
function Count-RunningRuns {
    # Each run shows up as two python.exe processes (the venv launcher and the
    # interpreter it starts), so count distinct run names, not processes.
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -like '*train_cv.py*' } |
      ForEach-Object { if ($_.CommandLine -match '--name\s+(\S+)') { $Matches[1] } } |
      Sort-Object -Unique).Count
}

foreach ($name in $batches[$Batch].Keys) {
    while ($Parallel -gt 0 -and (Count-RunningRuns) -ge $Parallel) { Start-Sleep -Seconds 30 }
    $options = $batches[$Batch][$name]
    $command = "`$Host.UI.RawUI.WindowTitle = 'train: $name'; " +
               "Set-Location '$root'; `$env:PYTHONUNBUFFERED = '1'; " +
               "uv run scripts/train_cv.py --name $name $options $common; " +
               "Write-Host ''; Write-Host 'Run $name finished. This window can be closed.' -ForegroundColor Green"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $command
    Write-Host "Started $name ($options)"
    if ($Parallel -gt 0) { Start-Sleep -Seconds 20 }  # let it start before counting
}
Write-Host "Logs: $root\runs\logs   Reports: $root\runs\<date>_<name>\report.md"
