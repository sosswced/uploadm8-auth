@echo off
setlocal ENABLEDELAYEDEXPANSION

REM CMD launcher for tools\run_internal_full_upload_tests.ps1
REM Usage:
REM   set VIDEO=C:\path\video.mp4
REM   set MAP=C:\path\video.map
REM   tools\run_internal_full_upload_tests.cmd

if "%VIDEO%"=="" (
  echo ERROR: VIDEO env var is not set.
  echo Example: set VIDEO=C:\Users\Earl\Videos\20250224_0073_CAM_EVNT.MP4
  exit /b 1
)

if "%MAP%"=="" (
  echo WARN: MAP env var is not set. Trill/HUD telemetry paths will be reduced.
)

REM Force UTF-8 output path so Unicode/emoji don't render as mojibake in cmd logs.
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0.."
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Failed to enter repo root.
  exit /b 1
)

set "PS_ARGS=-ExecutionPolicy Bypass -File tools\run_internal_full_upload_tests.ps1 -VideoPath "%VIDEO%" -RandomRuns 3 -SeedBase 1000 -LogLevel DEBUG"
if not "%MAP%"=="" (
  set "PS_ARGS=%PS_ARGS% -TelemetryMapPath "%MAP%""
)

echo Running internal full upload tests...
echo VIDEO=%VIDEO%
if not "%MAP%"=="" echo MAP=%MAP%
echo.

powershell %PS_ARGS%
set "RC=%ERRORLEVEL%"

popd >nul 2>&1
exit /b %RC%
