# EvoRoute - Flood-Resilient Road Priority Ranking System
# Native PowerShell Execution Script

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "            EVOROUTE: FLOOD-RESILIENT ROAD RANKING SYSTEM" -ForegroundColor Yellow
Write-Host "    Soft Computing Architecture: Mamdani FIS + Hybrid Genetic Algorithm" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found in your system PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ and ensure it is added to PATH."
    Read-Host "Press Enter to exit"
    exit 1
}

$pyVersion = python --version
Write-Host "[*] Python detected: $pyVersion" -ForegroundColor Green
Write-Host ""

# 2. Check and Install Dependencies
Write-Host "[*] Verifying required Python packages..." -ForegroundColor Yellow
$checkDeps = python -c "import networkx, numpy, scipy, pandas, matplotlib, folium, skfuzzy" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[*] Installing missing dependencies via pip..." -ForegroundColor Yellow
    pip install networkx numpy scipy pandas matplotlib folium scikit-fuzzy
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies. Check your internet connection." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[*] All dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "[*] All required dependencies are already installed." -ForegroundColor Green
}
Write-Host ""

# 3. Execute End-to-End Pipeline
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[*] Executing Full 4-Phase EvoRoute Pipeline..." -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

python run_pipeline.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Pipeline encountered an error during execution!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "[*] PIPELINE EXECUTION COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Generated Outputs:" -ForegroundColor Cyan
Write-Host "  - Interactive Map: results\maps\flood_resilience_map.html"
Write-Host "  - Figures:         results\figures\fig1_fuzzy_membership_and_surface.png"
Write-Host "                     results\figures\fig2_ga_convergence_and_ablation.png"
Write-Host "                     results\figures\fig3_benchmark_comparison.png"
Write-Host "  - CSV Tables:      results\tables\benchmark_results.csv"
Write-Host "                     results\tables\candidates_with_fuzzy_priority.csv"
Write-Host "                     results\tables\villages_svi.csv"
Write-Host "  - Specifications:  ASSUMPTIONS.md"
Write-Host ""

# 4. Optional Map Launch
$openMap = Read-Host "Do you want to open the interactive map in your browser now? [Y/N]"
if ($openMap -match "^[Yy]") {
    Start-Process "results\maps\flood_resilience_map.html"
}

Write-Host ""
Write-Host "Finished. Press Enter to exit..." -ForegroundColor Gray
Read-Host
