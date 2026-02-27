# ============================================================
# run_pipeline.ps1
# ============================================================
# PowerShell wrapper for Windows Task Scheduler.
# Task Scheduler calls this script, which activates the venv
# and runs the pipeline orchestrator.
#
# SETUP: Place this file at C:\BlueprintLLM\run_pipeline.ps1
#
# Arguments (passed from Task Scheduler):
#   -Mode       full | data-only | train-only | eval-only
#   -Force      Switch to force training even if data unchanged
#
# Examples:
#   .\run_pipeline.ps1 -Mode full
#   .\run_pipeline.ps1 -Mode data-only
#   .\run_pipeline.ps1 -Mode train-only -Force
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("full", "data-only", "train-only", "eval-only")]
    [string]$Mode,

    [switch]$Force
)

# --- Configuration ---
$ProjectRoot = "C:\BlueprintLLM"
$VenvPython  = "$ProjectRoot\venv\Scripts\python.exe"
$Script      = "$ProjectRoot\scripts\11_pipeline_orchestrator.py"
$LogDir      = "$ProjectRoot\logs"

# --- GPU pinning: use only GPU 0 (training GPU) ---
$env:CUDA_VISIBLE_DEVICES = "0"

# --- CUDA JIT cache: persist compiled kernels across runs ---
$cudaCache = "$ProjectRoot\.cuda_cache"
if (-not (Test-Path $cudaCache)) { New-Item -ItemType Directory -Path $cudaCache -Force | Out-Null }
$env:CUDA_CACHE_PATH = $cudaCache
$env:CUDA_CACHE_MAXSIZE = "4294967296"

# --- Ensure log directory exists ---
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# --- Build timestamp for this run ---
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog    = "$LogDir\scheduler_${Mode}_${Timestamp}.log"

# --- Build arguments ---
$PipelineArgs = @("$Script", "--$Mode")
if ($Force) {
    $PipelineArgs += "--force"
}

# --- Log start ---
"[$Timestamp] Starting pipeline: $Mode" | Out-File -FilePath $RunLog -Encoding utf8
"Command: $VenvPython $($PipelineArgs -join ' ')" | Out-File -FilePath $RunLog -Append -Encoding utf8
"" | Out-File -FilePath $RunLog -Append -Encoding utf8

# --- Run the pipeline ---
try {
    $process = Start-Process -FilePath $VenvPython `
        -ArgumentList $PipelineArgs `
        -WorkingDirectory $ProjectRoot `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput "$LogDir\stdout_${Timestamp}.log" `
        -RedirectStandardError  "$LogDir\stderr_${Timestamp}.log"

    $ExitCode = $process.ExitCode

    # Append stdout to run log
    if (Test-Path "$LogDir\stdout_${Timestamp}.log") {
        Get-Content "$LogDir\stdout_${Timestamp}.log" | Out-File -FilePath $RunLog -Append -Encoding utf8
    }

    # Append stderr if any
    $StderrContent = Get-Content "$LogDir\stderr_${Timestamp}.log" -ErrorAction SilentlyContinue
    if ($StderrContent) {
        "" | Out-File -FilePath $RunLog -Append -Encoding utf8
        "=== STDERR ===" | Out-File -FilePath $RunLog -Append -Encoding utf8
        $StderrContent | Out-File -FilePath $RunLog -Append -Encoding utf8
    }

    # Clean up temp files
    Remove-Item "$LogDir\stdout_${Timestamp}.log" -ErrorAction SilentlyContinue
    Remove-Item "$LogDir\stderr_${Timestamp}.log" -ErrorAction SilentlyContinue

    # Log result
    "" | Out-File -FilePath $RunLog -Append -Encoding utf8
    switch ($ExitCode) {
        0 { "RESULT: SUCCESS" | Out-File -FilePath $RunLog -Append -Encoding utf8 }
        1 { "RESULT: FAILURE (check errors above)" | Out-File -FilePath $RunLog -Append -Encoding utf8 }
        2 { "RESULT: SKIPPED (no new data to process)" | Out-File -FilePath $RunLog -Append -Encoding utf8 }
        default { "RESULT: UNKNOWN EXIT CODE $ExitCode" | Out-File -FilePath $RunLog -Append -Encoding utf8 }
    }

} catch {
    "EXCEPTION: $_" | Out-File -FilePath $RunLog -Append -Encoding utf8
    $ExitCode = 1
}

"Finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $RunLog -Append -Encoding utf8

exit $ExitCode
