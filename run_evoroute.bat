@echo off
setlocal EnableDelayedExpansion

title EvoRoute - Flood-Resilient Road Priority Ranking System

echo ======================================================================
echo             EVOROUTE: FLOOD-RESILIENT ROAD RANKING SYSTEM
echo     Soft Computing Architecture: Mamdani FIS + Hybrid Genetic Algorithm
echo ======================================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ and ensure it is added to PATH.
    echo.
    pause
    exit /b 1
)

echo [*] Python detected:
python --version
echo.

:: 2. Check and install core dependencies
echo [*] Verifying required Python packages...
python -c "import networkx, numpy, scipy, pandas, matplotlib, folium, skfuzzy" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [*] Installing missing dependencies via pip...
    pip install networkx numpy scipy pandas matplotlib folium scikit-fuzzy
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies. Please check your internet connection.
        pause
        exit /b 1
    )
    echo [*] All dependencies installed successfully.
) else (
    echo [*] All required dependencies are already installed.
)
echo.

:: 3. Run the end-to-end 4-Phase pipeline
echo ======================================================================
echo [*] Executing Full 4-Phase EvoRoute Pipeline...
echo ======================================================================
echo.

python run_pipeline.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline encountered an error during execution!
    echo Check the error messages above for details.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ======================================================================
echo [*] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo Generated Outputs:
echo   - Interactive Map: results\maps\flood_resilience_map.html
echo   - Figures:         results\figures\fig1_fuzzy_membership_and_surface.png
echo                      results\figures\fig2_ga_convergence_and_ablation.png
echo                      results\figures\fig3_benchmark_comparison.png
echo   - CSV Tables:      results\tables\benchmark_results.csv
echo                      results\tables\candidates_with_fuzzy_priority.csv
echo                      results\tables\villages_svi.csv
echo   - Specifications:  ASSUMPTIONS.md
echo.

:: 4. Ask user if they wish to open the interactive map
set /p OPEN_MAP="Do you want to open the interactive map in your browser now? [Y/N]: "
if /i "%OPEN_MAP%"=="Y" (
    start "" "results\maps\flood_resilience_map.html"
)

echo.
echo Finished. Press any key to exit...
pause >nul
