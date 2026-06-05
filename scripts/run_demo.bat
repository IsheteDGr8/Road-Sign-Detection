@echo off
REM Run SignDetector from the project root (so data/ is found).
cd /d "%~dp0\.."

set EXE=build\Debug\SignDetector.exe
if not exist "%EXE%" (
    echo Build first: open build\SignDetector.sln in Visual Studio and press F7.
    exit /b 1
)

echo Running %EXE%
echo Press any key through still-image tests; ESC during video to skip ahead.
"%EXE%"
