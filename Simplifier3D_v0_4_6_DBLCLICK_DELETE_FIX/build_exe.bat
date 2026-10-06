@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Simplifier3D v0.2 - lightweight build
echo   Uses the Python/PySide6 already installed on this PC
echo ============================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

echo [1/4] Checking imports...
%PY% -c "import PySide6, matplotlib, geometry3d_core, geometry3d_module; print('Imports: OK')"
if errorlevel 1 goto :error

echo [2/4] Checking PyInstaller...
%PY% -m PyInstaller --version
if errorlevel 1 (
    echo PyInstaller is not installed in this Python environment.
    echo Install once with:
    echo   %PY% -m pip install pyinstaller
    pause
    exit /b 1
)

echo [3/4] Cleaning old temporary build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist Simplifier3D.spec del /q Simplifier3D.spec

echo [4/4] Building...
%PY% -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --windowed ^
    --onedir ^
    --name Simplifier3D ^
    --hidden-import geometry3d_core ^
    --hidden-import geometry3d_module ^
    --add-data "geometry_config.json;." ^
    --add-data "model_data;model_data" ^
    rdp_simplifier26.py
if errorlevel 1 goto :error

if exist build rmdir /s /q build
if exist Simplifier3D.spec del /q Simplifier3D.spec

echo.
echo BUILD COMPLETE:
echo   %CD%\dist\Simplifier3D\Simplifier3D.exe
echo.
pause
exit /b 0

:error
echo.
echo BUILD FAILED.
pause
exit /b 1
