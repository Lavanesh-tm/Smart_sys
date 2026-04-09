@echo off
setlocal
REM Edit this line if your folder is not exactly D:\miniconda_new
set "MINICONDA_ROOT=D:\miniconda_new"

if not exist "%MINICONDA_ROOT%\Scripts\conda.exe" (
    echo [ERROR] Not found: "%MINICONDA_ROOT%\Scripts\conda.exe"
    echo Edit MINICONDA_ROOT in this file to your full path, then run again.
    exit /b 1
)

echo Using: "%MINICONDA_ROOT%\Scripts\conda.exe"
"%MINICONDA_ROOT%\Scripts\conda.exe" init powershell
if errorlevel 1 exit /b 1

echo.
echo Done. Next: open your PowerShell profile in Notepad and remove any old
echo "conda initialize" block that points to anaconda3 — keep only Miniconda.
echo Then restart Cursor.
pause
