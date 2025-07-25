
#!/bin/bash

set -e

echo "Detecting package manager..."

if command -v apt &> /dev/null; then
    PM="apt"
    sudo apt update
    sudo apt install -y git cmake build-essential python3 python3-pip libopenblas-dev curl
elif command -v dnf &> /dev/null; then
    PM="dnf"
    sudo dnf install -y git cmake make gcc gcc-c++ python3 python3-pip openblas-devel curl
elif command -v yum &> /dev/null; then
    PM="yum"
    sudo yum install -y git cmake make gcc gcc-c++ python3 python3-pip openblas-devel curl
elif command -v pacman &> /dev/null; then
    PM="pacman"
    sudo pacman -Sy --noconfirm git cmake base-devel python python-pip openblas curl
else
    echo "Unsupported package manager. Please install git, cmake, build tools, python3, pip, and OpenBLAS manually."
    exit 1
fi

echo "Cloning llama.cpp repository..."
if [ ! -d llama.cpp ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
else
    echo "llama.cpp directory already exists. Pulling latest changes..."
    cd llama.cpp
    git pull
    cd ..
fi

echo "Building llama.cpp..."
cd llama.cpp
make

echo "llama.cpp built successfully!"

# Optional: Build Python bindings
echo "Installing Python bindings for llama.cpp (optional)..."
cd python
pip3 install -r requirements.txt
pip3 install .

cd ..
cd ..

curl -fSsL https://ollama.com/install.sh | sh

echo "Setup complete!"
pip3 install -r requirements.txt
streamlit run setup.py
