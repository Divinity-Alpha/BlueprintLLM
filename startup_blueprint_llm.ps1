# ============================================================
# startup_blueprint_llm.ps1
# ============================================================
# Runs automatically at Windows logon (30 sec delay).
# Verifies the entire Blueprint LLM environment is healthy,
# fixes what it can, and flags what it can't.
#
# What it checks:
#   1. Python venv exists and works
#   2. All required packages installed
#   3. CUDA / GPU available
#   4. Hugging Face login valid
#   5. All directories exist (creates missing ones)
#   6. All scripts present
#   7. Scheduled tasks enabled (re-enables if disabled)
#   8. Model status and training history
#   9. Pending exports in inbox
#  10. Kicks off data processing if exports are waiting
# ============================================================

$ProjectRoot = "C:\BlueprintLLM"
$VenvPython  = "$ProjectRoot\venv\Scripts\python.exe"
$LogFile     = "$ProjectRoot\logs\startup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# GPU pinning: use only GPU 0 (training GPU)
$env:CUDA_VISIBLE_DEVICES = "0"

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    $line | Out-File -FilePath $LogFile -Append -Encoding utf8
}

# Ensure log directory
if (-not (Test-Path "$ProjectRoot\logs")) {
    New-Item -ItemType Directory -Path "$ProjectRoot\logs" -Force | Out-Null
}

Log "============================================"
Log "  Blueprint LLM Startup Check"
Log "============================================"
Log "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Log "Project: $ProjectRoot"

$allGood = $true

# --- Check 1: Project root ---
if (-not (Test-Path $ProjectRoot)) {
    Log "[FAIL] Project root not found: $ProjectRoot"
    exit 1
}
Log "[OK] Project root exists"

# --- Check 2: Python venv ---
if (-not (Test-Path $VenvPython)) {
    Log "[FAIL] Python venv not found: $VenvPython"
    $allGood = $false
} else {
    $pyVersion = & $VenvPython --version 2>&1
    Log "[OK] Python: $pyVersion"
}

# --- Check 3: Key packages ---
if (Test-Path $VenvPython) {
    $packages = @("torch", "transformers", "peft", "trl", "bitsandbytes")
    foreach ($pkg in $packages) {
        $result = & $VenvPython -c "import $pkg; print($pkg.__version__)" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log "[OK] $pkg $result"
        } else {
            Log "[FAIL] $pkg not installed"
            $allGood = $false
        }
    }
}

# --- Check 4: CUDA / GPU ---
if (Test-Path $VenvPython) {
    $cudaCheck = & $VenvPython -c @"
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    vram = props.total_memory / 1024**3
    print(f'{torch.cuda.get_device_name(0)} | {vram:.1f} GB | Compute {cap[0]}.{cap[1]}')
else:
    print('NO_CUDA')
"@ 2>&1
    if ($cudaCheck -match "NO_CUDA") {
        Log "[FAIL] CUDA not available. GPU training will not work."
        $allGood = $false
    } else {
        Log "[OK] GPU: $cudaCheck"
    }
}

# --- Check 5: Hugging Face token ---
if (Test-Path $VenvPython) {
    $hfCheck = & $VenvPython -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami()['name'])" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Log "[OK] HuggingFace: logged in as $hfCheck"
    } else {
        Log "[WARN] HuggingFace login not found or expired"
        Log "       Cached models still work, but can't download new ones."
        Log "       To fix: python -c `"from huggingface_hub import login; login(token=input('Token: '))`""
    }
}

# --- Check 6: Directories ---
$dirs = @(
    "raw-data\clipboard-exports",
    "raw-data\clipboard-exports\processed",
    "cleaned-data\parsed-blueprints",
    "datasets",
    "models",
    "results",
    "logs",
    "scripts\utils"
)
$fixedDirs = 0
foreach ($d in $dirs) {
    $fullPath = Join-Path $ProjectRoot $d
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        $fixedDirs++
    }
}
if ($fixedDirs -gt 0) {
    Log "[FIXED] Created $fixedDirs missing directories"
} else {
    Log "[OK] All directories present"
}

# --- Check 7: Scripts ---
$scripts = @(
    "scripts\01_analyze_blueprint_clipboard.py",
    "scripts\04_train_blueprint_lora.py",
    "scripts\05_auto_translate_export.py",
    "scripts\06_validate_dsl.py",
    "scripts\09_evaluate_model.py",
    "scripts\11_pipeline_orchestrator.py",
    "scripts\utils\dsl_parser.py",
    "scripts\utils\blueprint_patterns.py",
    "run_pipeline.ps1"
)
$missing = @()
foreach ($s in $scripts) {
    if (-not (Test-Path (Join-Path $ProjectRoot $s))) {
        $missing += $s
    }
}
if ($missing.Count -gt 0) {
    Log "[FAIL] Missing $($missing.Count) scripts: $($missing -join ', ')"
    $allGood = $false
} else {
    Log "[OK] All scripts present"
}

# --- Check 8: Scheduled tasks ---
$tasks = @(
    "BlueprintLLM - Data Processing",
    "BlueprintLLM - Nightly Training",
    "BlueprintLLM - Weekly Evaluation"
)
foreach ($t in $tasks) {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if ($task) {
        if ($task.State -eq "Disabled") {
            Enable-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
            Log "[FIXED] Re-enabled: $t"
        } else {
            Log "[OK] Task active: $t (next run: $($task.Triggers[0]))"
        }
    } else {
        Log "[WARN] Task missing: $t"
        Log "       Run: .\setup_scheduled_tasks.ps1 (as Admin)"
    }
}

# --- Check 9: Model status ---
$models = Get-ChildItem "$ProjectRoot\models" -Directory -Filter "blueprint-lora-v*" -ErrorAction SilentlyContinue | Sort-Object Name
if ($models) {
    $latest = $models[-1].Name
    Log "[OK] Latest model: $latest ($($models.Count) version(s) total)"
    $stateFile = "$ProjectRoot\.pipeline_state.json"
    if (Test-Path $stateFile) {
        $state = Get-Content $stateFile -Raw | ConvertFrom-Json
        Log "     Last run: $($state.last_run)"
    }
} else {
    Log "[INFO] No trained models yet. First nightly run will create v1."
}

# --- Check 10: Pending exports ---
$pending = @()
$pending += Get-ChildItem "$ProjectRoot\raw-data\clipboard-exports\*.txt" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch "\.(dsl|analysis)" }
$pendingCount = $pending.Count
if ($pendingCount -gt 0) {
    Log "[INFO] $pendingCount export(s) waiting to be processed"
} else {
    Log "[INFO] No pending exports. Drop .txt files in raw-data\clipboard-exports\"
}

# --- Summary ---
Log ""
Log "============================================"
if ($allGood) {
    Log "  ALL CHECKS PASSED - System is ready"
    Log "============================================"
    Log ""
    Log "  Scheduled tasks handle everything automatically:"
    Log "    Every 2 hrs:  Process new exports"
    Log "    2:00 AM:      Train + evaluate"
    Log "    Sunday 10 AM: Weekly progress report"
    Log ""
    Log "  Your only job: drop Blueprint .txt files in"
    Log "    $ProjectRoot\raw-data\clipboard-exports\"
} else {
    Log "  SOME CHECKS FAILED - Review [FAIL] items above"
    Log "============================================"
}

# --- Auto-process pending exports ---
if ($allGood -and $pendingCount -gt 0) {
    Log ""
    Log "Processing $pendingCount pending export(s)..."
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-ExecutionPolicy Bypass -File `"$ProjectRoot\run_pipeline.ps1`" -Mode data-only" `
        -WorkingDirectory $ProjectRoot `
        -NoNewWindow
    Log "Data pipeline started."
}

Log ""
Log "Startup check complete."
Log "Log: $LogFile"
