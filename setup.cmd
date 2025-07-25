@echo off
SETLOCAL EnableDelayedExpansion

echo Detecting package manager...

:: Windows doesn't have a unified package manager by default.
:: Recommend using Chocolatey or manual install.
where choco >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    SET PM=choco
    echo Using Chocolatey...
    choco install -y git cmake python openblas curl
) ELSE (
    echo No package manager detected. Please install the following manually:
    echo - Git
    echo - CMake
    echo - Python 3 and pip
    echo - OpenBLAS (optional but recommended)
    echo - curl (or ensure it's in PATH)
    pause
)

:: Clone llama.cpp repo
IF NOT EXIST llama.cpp (
    echo Cloning llama.cpp repository...
    git clone https://github.com/ggml-org/llama.cpp.git
) ELSE (
    echo llama.cpp directory already exists. Pulling latest changes...
    pushd llama.cpp
    git pull
    popd
)

:: Build llama.cpp
echo Building llama.cpp...
cd llama.cpp

IF NOT EXIST build (
    mkdir build
)
cd build

cmake .. -DLLAMA_CURL=OFF
IF %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed. Exiting...
    exit /b 1
)

cmake --build . --config Release
IF %ERRORLEVEL% NEQ 0 (
    echo Build failed. Exiting...
    exit /b 1
)

cd ..

:: Python bindings
echo Installing Python bindings for llama.cpp (optional)...
cd python

pip install -r requirements.txt
pip install .

cd ..

:: Ollama installation (Windows support is limited)
curl -L O https://release-assets.githubusercontent.com/github-production-release-asset/658928958/e8384a9d-8b1e-4742-9400-7a0ce2adb947?sp=r&sv=2018-11-09&sr=b&spr=https&se=2025-07-24T05%3A38%3A35Z&rscd=attachment%3B+filename%3DOllamaSetup.exe&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2025-07-24T04%3A38%3A11Z&ske=2025-07-24T05%3A38%3A35Z&sks=b&skv=2018-11-09&sig=SsdxrkUstcNXargorol3Vy1TGZ0LfAjX3LElROTXzzc%3D&jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc1MzMzMzQ5NCwibmJmIjoxNzUzMzMzMTk0LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.acfDSRqgrUatsIOZwZIrx7-vWXw2OD-dTTbSsnCiMPw&response-content-disposition=attachment%3B%20filename%3DOllamaSetup.exe&response-content-type=application%2Foctet-stream
echo Setup complete!
pause
