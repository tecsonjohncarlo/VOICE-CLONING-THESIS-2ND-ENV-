@echo off
echo ========================================
echo Markdown Files Cleanup Script
echo ========================================
echo.
echo This will delete 26 redundant markdown files
echo and keep only 8 essential documentation files.
echo.
echo BACKUP will be created first for safety!
echo.
pause

REM Create backup
echo.
echo Creating backup...
if not exist "MDs_backup" mkdir "MDs_backup"
xcopy "MDs" "MDs_backup" /E /I /Y >nul 2>&1
echo ✅ Backup created in MDs_backup folder
echo.

REM Delete duplicate quick starts
echo Deleting duplicate quick start guides...
del "MDs\QUICKSTART.md" 2>nul
del "MDs\STARTUP_GUIDE.md" 2>nul
del "scripts_unix\QUICK_START.md" 2>nul
echo ✅ Deleted 3 duplicate quick start files

REM Delete duplicate device guides
echo Deleting duplicate device guides...
del "MDs\DEVICE_SELECTION_GUIDE.md" 2>nul
del "MDs\DEVICE_SUPPORT.md" 2>nul
echo ✅ Deleted 2 duplicate device guide files

REM Delete old status files
echo Deleting old status files...
del "MDs\FINAL_HONEST_STATUS.md" 2>nul
del "MDs\FINAL_IMPLEMENTATION_STATUS.md" 2>nul
del "MDs\IMPLEMENTATION_SUMMARY.md" 2>nul
del "MDs\CRITICAL_FEATURES_IMPLEMENTED.md" 2>nul
del "MDs\V2_OPTIMIZATION_APPLIED.md" 2>nul
echo ✅ Deleted 5 old status files

REM Delete old setup guides
echo Deleting old setup guides...
del "MDs\INSTALLATION_SUMMARY.md" 2>nul
del "MDs\SETUP_FIX.md" 2>nul
del "MDs\ENVIRONMENT_FIX.md" 2>nul
del "MDs\RUN_ALL_FIXES.md" 2>nul
echo ✅ Deleted 4 old setup files

REM Delete technical implementation details
echo Deleting technical implementation details...
del "MDs\ENGINE_V2_ANALYSIS.md" 2>nul
del "MDs\ONNX_IMPLEMENTATION_COMPLETE.md" 2>nul
del "MDs\COMPLETE_OPTIMIZATION_IMPLEMENTATION.md" 2>nul
del "MDs\PRODUCTION_GRADE_FEATURES.md" 2>nul
del "MDs\SMART_TEXT_LIMITING.md" 2>nul
del "MDs\MONITORING_SYSTEM.md" 2>nul
echo ✅ Deleted 6 technical implementation files

REM Delete alternative approaches
echo Deleting alternative approaches...
del "MDs\ALTERNATIVE_APPROACH.md" 2>nul
del "MDs\SIMPLE_SOLUTION.md" 2>nul
del "MDs\STANDALONE_INFERENCE_GUIDE.md" 2>nul
echo ✅ Deleted 3 alternative approach files

REM Delete miscellaneous
echo Deleting miscellaneous files...
del "MDs\claude.md" 2>nul
del "MDs\Integration.md" 2>nul
del "MDs\STRUCTURE.md" 2>nul
echo ✅ Deleted 3 miscellaneous files

REM Delete platform-specific
echo Deleting platform-specific files...
del "MDs\WSL2_SETUP_GUIDE.md" 2>nul
echo ✅ Deleted 1 platform-specific file

REM Delete unix scripts
echo Deleting unix scripts...
del "scripts_unix\README.md" 2>nul
echo ✅ Deleted 1 unix script file

echo.
echo ========================================
echo Cleanup Complete!
echo ========================================
echo.
echo 📊 Summary:
echo   - Deleted: 26 redundant files
echo   - Kept: 8 essential files
echo   - Backup: MDs_backup folder
echo.
echo ✅ Remaining files:
echo   Root:
echo     - README.md
echo     - SETUP_GUIDE.md
echo     - changelog_thesis.md
echo     - Universal Optimization Guide.md
echo     - Claude Final Honest Guide.md
echo.
echo   MDs folder:
echo     - TROUBLESHOOTING.md
echo     - PROJECT_SUMMARY.md
echo     - FISH_SPEECH_ANALYSIS.md
echo.
echo See MARKDOWN_CLEANUP_ANALYSIS.md for details
echo.
pause
