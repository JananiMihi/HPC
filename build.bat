@echo off
REM Build Script for Phase 1: Serial Baseline
REM This script attempts to compile using available compilers

echo.
echo === Phase 1: Serial Baseline - Build Script ===
echo.

setlocal enabledelayedexpansion

set EXECUTABLE=phase1_serial.exe
set SOURCE_FILES=phase1_serial.cpp

REM Try MSVC Compiler
where cl.exe >nul 2>nul
if !errorlevel! equ 0 (
    echo Found MSVC compiler (cl.exe)
    echo Compiling...
    cl /O2 /std:c++latest /EHsc %SOURCE_FILES% /Fe%EXECUTABLE%
    if !errorlevel! equ 0 (
        echo Build successful: %EXECUTABLE%
        echo.
        echo Run with: %EXECUTABLE%
    ) else (
        echo MSVC compilation failed
        goto try_gcc
    )
    goto end
)

:try_gcc
REM Try MinGW/GCC Compiler
where g++.exe >nul 2>nul
if !errorlevel! equ 0 (
    echo Found GCC compiler (g++.exe)
    echo Compiling...
    g++ -std=c++17 -O2 -o %EXECUTABLE% %SOURCE_FILES%
    if !errorlevel! equ 0 (
        echo Build successful: %EXECUTABLE%
        echo.
        echo Run with: %EXECUTABLE%
    ) else (
        echo GCC compilation failed
        goto try_clang
    )
    goto end
)

:try_clang
REM Try Clang Compiler
where clang++.exe >nul 2>nul
if !errorlevel! equ 0 (
    echo Found Clang compiler (clang++.exe)
    echo Compiling...
    clang++ -std=c++17 -O2 -o %EXECUTABLE% %SOURCE_FILES%
    if !errorlevel! equ 0 (
        echo Build successful: %EXECUTABLE%
        echo.
        echo Run with: %EXECUTABLE%
    ) else (
        echo Clang compilation failed
        goto no_compiler
    )
    goto end
)

:no_compiler
echo.
echo ERROR: No C++ compiler found!
echo.
echo Please install one of the following:
echo  1. MSVC (Visual Studio C++ build tools)
echo  2. MinGW (https://www.mingw-w64.org/)
echo  3. Clang (https://clang.llvm.org/)
echo.
goto end

:end
echo.
pause
