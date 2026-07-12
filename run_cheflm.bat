@echo off
setlocal enabledelayedexpansion
title ChefLM - Local Runner
color 0A

echo ============================================
echo   ChefLM - Local Setup and Chat
echo ============================================
echo.

REM ---------------------------------------------------------------
REM 1. Check for Python
REM ---------------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on your PATH.
    echo Install Python 3.9+ from https://www.python.org/downloads/
    echo and make sure "Add python.exe to PATH" is checked during install.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 2. Get the repo (clone if this .bat is run outside the repo)
REM ---------------------------------------------------------------
if exist "chef\__main__.py" (
    echo [OK] Already inside the chefLM repo folder.
) else (
    echo chefLM repo not found in current folder.
    where git >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Git was not found on your PATH, and the repo isn't here.
        echo Install Git from https://git-scm.com/download/win, or manually
        echo download/extract the repo from:
        echo   https://github.com/BT-Rajan/chefLM
        echo and place this .bat file in the repo's root folder.
        pause
        exit /b 1
    )
    echo Cloning github.com/BT-Rajan/chefLM ...
    git clone https://github.com/BT-Rajan/chefLM.git
    if errorlevel 1 (
        echo [ERROR] git clone failed. Check your internet connection.
        pause
        exit /b 1
    )
    cd chefLM
)

if not exist "webui\installer.py" (
    echo [ERROR] webui\installer.py not found in the repo folder.
    echo Make sure you copied the webui\ folder in alongside this .bat
    echo file - see the setup instructions.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------
REM 3. Hand off to the interactive browser installer
REM ---------------------------------------------------------------
echo.
echo Launching the setup page in your browser...
echo Leave this window open - it runs the install/train/chat process.
echo Close it (or press Ctrl+C) to stop everything.
echo.
python webui\installer.py

echo.
echo Stopped. Press any key to close.
pause >nul
