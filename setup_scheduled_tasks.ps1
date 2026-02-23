# ============================================================
# setup_scheduled_tasks.ps1
# ============================================================
# Run this ONCE (as Administrator) to create all the Windows
# Task Scheduler tasks for the Blueprint LLM pipeline.
#
# USAGE:
#   1. Right-click PowerShell → "Run as administrator"
#   2. cd C:\BlueprintLLM
#   3. .\setup_scheduled_tasks.ps1
#
# WHAT IT CREATES:
#   Task 1: "BlueprintLLM - Data Processing" (every 2 hours)
#           Analyzes new exports, validates, merges data
#
#   Task 2: "BlueprintLLM - Nightly Training" (2:00 AM daily)
#           Full pipeline: data → train → evaluate
#
#   Task 3: "BlueprintLLM - Weekly Evaluation" (Sunday 10:00 AM)
#           Evaluate the latest model with full test suite
#
# TO MODIFY AFTER CREATION:
#   Open Task Scheduler (taskschd.msc) → Task Scheduler Library
#   → Find the tasks and edit triggers, schedules, etc.
#
# TO REMOVE ALL TASKS:
#   .\setup_scheduled_tasks.ps1 -Remove
# ============================================================

param(
    [switch]$Remove
)

$ProjectRoot = "C:\BlueprintLLM"
$PowerShell  = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
$TaskPrefix  = "BlueprintLLM"

# --- Check admin ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator." -ForegroundColor Red
    Write-Host "Right-click PowerShell → 'Run as administrator'" -ForegroundColor Yellow
    exit 1
}

# --- Remove mode ---
if ($Remove) {
    Write-Host "Removing all BlueprintLLM scheduled tasks..." -ForegroundColor Yellow

    @("$TaskPrefix - Data Processing",
      "$TaskPrefix - Nightly Training",
      "$TaskPrefix - Weekly Evaluation") | ForEach-Object {
        $existing = Get-ScheduledTask -TaskName $_ -ErrorAction SilentlyContinue
        if ($existing) {
            Unregister-ScheduledTask -TaskName $_ -Confirm:$false
            Write-Host "  Removed: $_" -ForegroundColor Green
        } else {
            Write-Host "  Not found: $_" -ForegroundColor Gray
        }
    }

    Write-Host "`nDone. All tasks removed." -ForegroundColor Green
    exit 0
}

# --- Verify project exists ---
if (-not (Test-Path "$ProjectRoot\scripts\11_pipeline_orchestrator.py")) {
    Write-Host "ERROR: Project not found at $ProjectRoot" -ForegroundColor Red
    Write-Host "Make sure BlueprintLLM project is set up there first." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "$ProjectRoot\run_pipeline.ps1")) {
    Write-Host "ERROR: run_pipeline.ps1 not found at $ProjectRoot" -ForegroundColor Red
    Write-Host "Copy run_pipeline.ps1 to $ProjectRoot first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Setting up BlueprintLLM scheduled tasks..." -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot" -ForegroundColor Gray
Write-Host ""

# ============================================================
# TASK 1: Data Processing (every 2 hours)
# ============================================================
# This is lightweight — just analyzes new exports, validates
# DSL files, and merges training data. No GPU needed.
# Runs frequently so data is always ready when training kicks off.

$TaskName1 = "$TaskPrefix - Data Processing"
Write-Host "Creating: $TaskName1" -ForegroundColor Yellow

$Action1 = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument "-ExecutionPolicy Bypass -File `"$ProjectRoot\run_pipeline.ps1`" -Mode data-only" `
    -WorkingDirectory $ProjectRoot

$Trigger1 = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date -Hour 8 -Minute 0 -Second 0) `
    -RepetitionInterval (New-TimeSpan -Hours 2) `
    -RepetitionDuration (New-TimeSpan -Days 365)

$Settings1 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

# Remove existing if present
$existing = Get-ScheduledTask -TaskName $TaskName1 -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName1 -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName1 `
    -Action $Action1 `
    -Trigger $Trigger1 `
    -Settings $Settings1 `
    -Description "Analyzes new Blueprint exports, validates DSL files, merges training data. Runs every 2 hours." `
    -RunLevel Highest | Out-Null

Write-Host "  Created: Runs every 2 hours starting 8:00 AM" -ForegroundColor Green


# ============================================================
# TASK 2: Nightly Training (2:00 AM daily)
# ============================================================
# Full pipeline: data processing → training → evaluation.
# Runs at 2 AM so your GPU is free and it doesn't interrupt work.
# Automatically skips training if no new data since last run.

$TaskName2 = "$TaskPrefix - Nightly Training"
Write-Host "Creating: $TaskName2" -ForegroundColor Yellow

$Action2 = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument "-ExecutionPolicy Bypass -File `"$ProjectRoot\run_pipeline.ps1`" -Mode full" `
    -WorkingDirectory $ProjectRoot

$Trigger2 = New-ScheduledTaskTrigger -Daily -At "2:00AM"

$Settings2 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -MultipleInstances IgnoreNew

$existing = Get-ScheduledTask -TaskName $TaskName2 -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName2 -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName2 `
    -Action $Action2 `
    -Trigger $Trigger2 `
    -Settings $Settings2 `
    -Description "Full pipeline: analyze data, train model, evaluate. Runs nightly at 2 AM. Skips if no new data." `
    -RunLevel Highest | Out-Null

Write-Host "  Created: Runs daily at 2:00 AM" -ForegroundColor Green


# ============================================================
# TASK 3: Weekly Evaluation (Sunday 10:00 AM)
# ============================================================
# Just runs the evaluation suite against the latest model.
# Good for tracking progress over time without retraining.

$TaskName3 = "$TaskPrefix - Weekly Evaluation"
Write-Host "Creating: $TaskName3" -ForegroundColor Yellow

$Action3 = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument "-ExecutionPolicy Bypass -File `"$ProjectRoot\run_pipeline.ps1`" -Mode eval-only" `
    -WorkingDirectory $ProjectRoot

$Trigger3 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "10:00AM"

$Settings3 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -MultipleInstances IgnoreNew

$existing = Get-ScheduledTask -TaskName $TaskName3 -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName3 -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName3 `
    -Action $Action3 `
    -Trigger $Trigger3 `
    -Settings $Settings3 `
    -Description "Evaluates the latest model against the full test suite. Runs weekly on Sunday." `
    -RunLevel Highest | Out-Null

Write-Host "  Created: Runs Sundays at 10:00 AM" -ForegroundColor Green


# ============================================================
# SUMMARY
# ============================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  All tasks created successfully!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Schedule:" -ForegroundColor White
Write-Host "  Every 2 hours:  Data processing (analyze, validate, merge)" -ForegroundColor Gray
Write-Host "  Daily 2:00 AM:  Full training pipeline (train + eval)" -ForegroundColor Gray
Write-Host "  Sunday 10:00 AM: Weekly evaluation report" -ForegroundColor Gray
Write-Host ""
Write-Host "Your workflow:" -ForegroundColor White
Write-Host "  1. Build Blueprints in UE5" -ForegroundColor Gray
Write-Host "  2. Ctrl+A → Ctrl+C → paste into .txt files" -ForegroundColor Gray
Write-Host "  3. Save .txt files to: $ProjectRoot\raw-data\clipboard-exports\" -ForegroundColor Gray
Write-Host "  4. Write clean DSL files (optional, for quality)" -ForegroundColor Gray
Write-Host "  5. Sleep. The pipeline handles the rest." -ForegroundColor Gray
Write-Host ""
Write-Host "View tasks: Open Task Scheduler (taskschd.msc)" -ForegroundColor Yellow
Write-Host "View logs:  $ProjectRoot\logs\" -ForegroundColor Yellow
Write-Host "Remove all: .\setup_scheduled_tasks.ps1 -Remove" -ForegroundColor Yellow
Write-Host ""
Write-Host "Test it now (dry run):" -ForegroundColor White
Write-Host "  .\run_pipeline.ps1 -Mode full" -ForegroundColor Cyan
