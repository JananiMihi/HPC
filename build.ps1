# Build Script for Phase 1: Serial Baseline (PowerShell)
# This script attempts to compile using available compilers

Write-Host ""
Write-Host "=== Phase 1: Serial Baseline - Build Script ===" -ForegroundColor Green
Write-Host ""

$EXECUTABLE = "phase1_serial.exe"
$SOURCE_FILES = @("phase1_serial.cpp")

# Check for MSVC Compiler
$msvc = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($msvc) {
    Write-Host "Found MSVC compiler (cl.exe)" -ForegroundColor Cyan
    Write-Host "Compiling..."
    & cl.exe /O2 /std:c++latest /EHsc $SOURCE_FILES /Fe$EXECUTABLE
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful: $EXECUTABLE" -ForegroundColor Green
        Write-Host ""
        Write-Host "Run with: .\$EXECUTABLE" -ForegroundColor Yellow
    } else {
        Write-Host "MSVC compilation failed" -ForegroundColor Red
    }
    exit
}

# Check for GCC Compiler
$gcc = Get-Command g++.exe -ErrorAction SilentlyContinue
if ($gcc) {
    Write-Host "Found GCC compiler (g++.exe)" -ForegroundColor Cyan
    Write-Host "Compiling..."
    & g++.exe -std=c++17 -O2 -o $EXECUTABLE $SOURCE_FILES
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful: $EXECUTABLE" -ForegroundColor Green
        Write-Host ""
        Write-Host "Run with: .\$EXECUTABLE" -ForegroundColor Yellow
    } else {
        Write-Host "GCC compilation failed" -ForegroundColor Red
    }
    exit
}

# Check for Clang Compiler
$clang = Get-Command clang++.exe -ErrorAction SilentlyContinue
if ($clang) {
    Write-Host "Found Clang compiler (clang++.exe)" -ForegroundColor Cyan
    Write-Host "Compiling..."
    & clang++.exe -std=c++17 -O2 -o $EXECUTABLE $SOURCE_FILES
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Build successful: $EXECUTABLE" -ForegroundColor Green
        Write-Host ""
        Write-Host "Run with: .\$EXECUTABLE" -ForegroundColor Yellow
    } else {
        Write-Host "Clang compilation failed" -ForegroundColor Red
    }
    exit
}

# No compiler found
Write-Host ""
Write-Host "ERROR: No C++ compiler found!" -ForegroundColor Red
Write-Host ""
Write-Host "Please install one of the following:" -ForegroundColor Yellow
Write-Host " 1. MSVC (Visual Studio C++ build tools)"
Write-Host " 2. MinGW (https://www.mingw-w64.org/)"
Write-Host " 3. Clang (https://clang.llvm.org/)"
Write-Host ""
