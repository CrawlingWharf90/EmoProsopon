@echo off
:: 1. Silently check if Python exists in the system PATH
python --version >nul 2>&1

:: 2. If the error level is not 0, Python is missing
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================================
    echo  ERROR: Python 3 is not installed or not in your PATH.
    echo ========================================================
    echo EmoProsopon requires Python 3.12+ to run.
    echo.
    echo Please download it here: https://www.python.org/downloads/
    echo.
    echo IMPORTANT: When installing, you MUST check the box that says:
    echo "Add Python to PATH" at the bottom of the installer window.
    echo.
    exit /b 1
)

:: 3. If Python exists, pass all arguments to the main script
python "%~dp0eop.py" %*