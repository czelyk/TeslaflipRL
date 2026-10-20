@echo off
setlocal
cd /d "%~dp0"
if not exist ".runtime\python.exe" (
  echo Once powershell -ExecutionPolicy Bypass -File setup.ps1 calistirin.
  exit /b 1
)
set "PATH=%~dp0.runtime;%~dp0.runtime\Library\bin;%~dp0.runtime\Scripts;%PATH%"
"%~dp0.runtime\python.exe" %*
exit /b %errorlevel%
