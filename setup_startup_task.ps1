# ============================================================
# setup_startup_task.ps1
# ============================================================
# Run ONCE as Administrator to register the startup health
# check in Windows Task Scheduler.
#
# USAGE:
#   Right-click PowerShell -> Run as administrator
#   cd C:\BlueprintLLM
#   .\setup_startup_task.ps1
#
# TO REMOVE:
#   .\setup_startup_task.ps1 -Remove
# ============================================================

param(
    [switch]$Remove
)

$TaskName    = "BlueprintLLM - Startup Check"
$ProjectRoot = "C:\BlueprintLLM"
$PowerShell  = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

# Check admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdmin) {
    Write-Host "ERROR: Must run as Administrator." -ForegroundColor Red
    Write-Host "Right-click PowerShell -> Run as administrator" -ForegroundColor Yellow
    exit 1
}

# Remove mode
if ($Remove) {
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed: $TaskName" -ForegroundColor Green
    } else {
        Write-Host "Not found: $TaskName" -ForegroundColor Gray
    }
    exit 0
}

# Verify startup script exists
if (-not (Test-Path "$ProjectRoot\startup_blueprint_llm.ps1")) {
    Write-Host "ERROR: startup_blueprint_llm.ps1 not found at $ProjectRoot" -ForegroundColor Red
    exit 1
}

Write-Host "Creating startup task..." -ForegroundColor Yellow

$Action = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ProjectRoot\startup_blueprint_llm.ps1`"" `
    -WorkingDirectory $ProjectRoot

# Trigger at logon with 30-second delay
$Trigger = New-ScheduledTaskTrigger -AtLogon
$Trigger.Delay = "PT30S"

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -MultipleInstances IgnoreNew

# Remove existing if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Verifies Blueprint LLM environment on boot and processes pending exports." `
    -RunLevel Highest | Out-Null

Write-Host ""
Write-Host "Created: $TaskName" -ForegroundColor Green
Write-Host ""
Write-Host "On every reboot/login, this will:" -ForegroundColor White
Write-Host "  1. Check Python, GPU, packages, HuggingFace" -ForegroundColor Gray
Write-Host "  2. Fix missing directories" -ForegroundColor Gray
Write-Host "  3. Re-enable any disabled scheduled tasks" -ForegroundColor Gray
Write-Host "  4. Process pending Blueprint exports" -ForegroundColor Gray
Write-Host "  5. Log results to C:\BlueprintLLM\logs\startup_*.log" -ForegroundColor Gray
Write-Host ""
Write-Host "You now have 4 scheduled tasks:" -ForegroundColor Yellow
Get-ScheduledTask -TaskName "BlueprintLLM*" | Format-Table TaskName, State -AutoSize
Write-Host ""
Write-Host "Test it right now (optional):" -ForegroundColor Yellow
Write-Host "  .\startup_blueprint_llm.ps1" -ForegroundColor Cyan
