import streamlit as st
import subprocess
import platform
import sys
import os
import re
import time

st.set_page_config(page_title="GPU Auto-Setup", page_icon=":rocket:", layout="centered")

# ---------- Utility Functions ----------

def run_command(cmd, shell=False, show_output=False):
    if isinstance(cmd, list):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd
    with st.spinner(f"Running: `{cmd_str}`"):
        try:
            result = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if show_output or result.returncode != 0:
                st.code(result.stdout + "\n" + result.stderr)
            if result.returncode != 0:
                st.error(f"Command failed: {cmd_str}")
            return result.stdout.strip(), result.returncode == 0
        except Exception as e:
            st.error(str(e))
            return "", False

def get_cuda_version():
    out, ok = run_command(['nvcc', '--version'])
    if ok:
        match = re.search(r'release (\d+\.\d+)', out)
        if match:
            return match.group(1)
    out, ok = run_command(['nvidia-smi'])
    if ok:
        match = re.search(r'CUDA Version: (\d+\.\d+)', out)
        if match:
            return match.group(1)
    return None

def is_cuda_installed():
    _, ok = run_command(['which', 'nvidia-smi'])
    return ok

def is_rocm_installed():
    _, ok = run_command(['which', 'rocm-smi'])
    return ok

def detect_os():
    os_name = platform.system().lower()
    if os_name == "windows":
        return "Windows"
    elif os_name == "linux":
        return "Linux"
    elif os_name == "darwin":
        return "macOS"
    else:
        return os_name.capitalize()

def check_amd_windows():
    if platform.system().lower() == "windows":
        try:
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "AMD" in result.stdout or "Radeon" in result.stdout or "Advanced Micro Devices" in result.stdout:
                st.error("🚫 <b>ROCm is NOT supported on Windows with AMD GPUs.</b><br>Please use Linux for ROCm support.", unsafe_allow_html=True)
                return True
        except Exception as e:
            st.warning(f"Could not check GPU on Windows: {e}")
    return False

# ---------- Installers ----------

def install_cuda_linux():
    st.info("🔧 <b>Installing CUDA (Linux)</b>", unsafe_allow_html=True)
    if run_command(['which', 'apt'])[1]:
        st.write("Detected <b>apt</b> package manager.", unsafe_allow_html=True)
        run_command(['sudo', 'apt', 'update'])
        run_command(['sudo', 'apt', 'install', '-y', 'nvidia-cuda-toolkit'])
    elif run_command(['which', 'dnf'])[1]:
        st.write("Detected <b>dnf</b> package manager.", unsafe_allow_html=True)
        run_command(['sudo', 'dnf', 'install', '-y', 'cuda'])
    elif run_command(['which', 'pacman'])[1]:
        st.write("Detected <b>pacman</b> package manager.", unsafe_allow_html=True)
        run_command(['sudo', 'pacman', '-S', '--noconfirm', 'cuda'])
    else:
        st.error("Unknown package manager. Please install CUDA manually from NVIDIA's website.")

def install_cuda_windows():
    st.info("🔧 <b>CUDA Installation (Windows)</b>", unsafe_allow_html=True)
    st.markdown("Automatic CUDA installation is not supported on Windows.<br>"
                "Please download and install CUDA from the official website, then click <b>Continue</b>.",
                unsafe_allow_html=True)
    st.markdown("[CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)")
    if st.button("Continue"):
        install_pytorch_cuda(get_cuda_version())

def install_rocm_linux():
    st.info("🔧 <b>Installing ROCm (Linux)</b>", unsafe_allow_html=True)
    if not os.path.exists("/etc/os-release"):
        st.error("Cannot detect Linux distribution. Please install ROCm manually.")
        return
    with open("/etc/os-release") as f:
        os_release = f.read().lower()
    # ROCm installer URLs for major distros
    rocm_pkgs = {
        "ubuntu": "https://repo.radeon.com/amdgpu-install/6.1.2/ubuntu/focal/amdgpu-install_6.1.60102-1_all.deb",
        "debian": "https://repo.radeon.com/amdgpu-install/6.1.2/ubuntu/focal/amdgpu-install_6.1.60102-1_all.deb",
        "rhel":   "https://repo.radeon.com/amdgpu-install/6.1.2/rhel/8.7/amdgpu-install-6.1.60102-1.el8.noarch.rpm",
        "centos": "https://repo.radeon.com/amdgpu-install/6.1.2/rhel/8.7/amdgpu-install-6.1.60102-1.el8.noarch.rpm",
        "fedora": "https://repo.radeon.com/amdgpu-install/6.1.2/fedora/37/amdgpu-install-6.1.60102-1.fc37.noarch.rpm",
        "sles":   "https://repo.radeon.com/amdgpu-install/6.1.2/sles/15.4/amdgpu-install-6.1.60102-1.sle15.noarch.rpm",
    }
    for distro, url in rocm_pkgs.items():
        if distro in os_release:
            st.write(f"Detected <b>{distro.title()}</b> system.", unsafe_allow_html=True)
            pkg_file = url.split('/')[-1]
            run_command(f"wget {url}", shell=True)
            if pkg_file.endswith('.deb'):
                run_command(['sudo', 'dpkg', '-i', pkg_file])
                run_command(['sudo', 'apt', 'update'])
            elif pkg_file.endswith('.rpm'):
                run_command(['sudo', 'rpm', '-ivh', pkg_file])
                if 'fedora' in distro:
                    run_command(['sudo', 'dnf', 'makecache'])
                elif 'sles' in distro:
                    run_command(['sudo', 'zypper', 'refresh'])
                else:
                    run_command(['sudo', 'yum', 'makecache'])
            run_command(['sudo', 'amdgpu-install', '--usecase=rocm', '-y'])
            return
    st.error("Automatic ROCm installation is only supported for Ubuntu, Debian, RHEL, CentOS, Fedora, and SLES. For other distributions, please follow the [official ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.1.2/how-to/amdgpu-install.html).")

def install_pytorch_rocm():
    st.info("🔧 <b>Installing PyTorch for ROCm</b>", unsafe_allow_html=True)
    run_command([
        sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 'torchvision', 'torchaudio',
        '--index-url', 'https://download.pytorch.org/whl/nightly/rocm6.4/'
    ], show_output=True)

def install_pytorch_cuda(cuda_version):
    st.info(f"🔧 <b>Installing PyTorch for CUDA {cuda_version}</b>", unsafe_allow_html=True)
    cuda_to_url = {
        "12.1": "https://download.pytorch.org/whl/cu121",
        "12.0": "https://download.pytorch.org/whl/cu120",
        "11.8": "https://download.pytorch.org/whl/cu118",
        "11.7": "https://download.pytorch.org/whl/cu117",
        "11.6": "https://download.pytorch.org/whl/cu116",
        "11.3": "https://download.pytorch.org/whl/cu113",
        "11.1": "https://download.pytorch.org/whl/cu111",
        "10.2": "https://download.pytorch.org/whl/cu102",
    }
    url = cuda_to_url.get(cuda_version)
    if url:
        run_command([
            sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio',
            '--index-url', url
        ], show_output=True)
    else:
        st.warning(f"CUDA version {cuda_version} is not directly supported by PyTorch wheels. Installing CPU-only version.")
        run_command([
            sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'
        ], show_output=True)

# ---------- Setup Functions ----------

def setup_nvidia():
    st.subheader("🟩 NVIDIA GPU Detected")
    if is_cuda_installed():
        st.success("CUDA detected!")
    else:
        if detect_os() == "Linux":
            install_cuda_linux()
        elif detect_os() == "Windows":
            install_cuda_windows()
        else:
            st.error("Unsupported OS for CUDA installation.")
    if is_cuda_installed():
        cuda_version = get_cuda_version()
        if cuda_version:
            st.success(f"Detected CUDA version: {cuda_version}")
            install_pytorch_cuda(cuda_version)
        else:
            st.warning("Could not detect CUDA version. Installing CPU-only PyTorch.")
            run_command([
                sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'
            ], show_output=True)
    # Set flags for NVIDIA: both fp16 and bf16 enabled
    with open('FLAGS.txt', 'w') as f:
        f.write('use_gpu=True\nfp16=True\nbf16=True\n')
    st.success("NVIDIA setup complete! 🚀")

def setup_amd():
    st.subheader("🟧 AMD GPU Detected")
    if is_rocm_installed():
        st.success("ROCm detected!")
    else:
        if detect_os() == "Linux":
            install_rocm_linux()
        else:
            st.error("ROCm is only supported on Linux.")
    if is_rocm_installed():
        install_pytorch_rocm()
    # Set flags for AMD: bf16 preferred, fp16 fallback
    with open('FLAGS.txt', 'w') as f:
        f.write('use_gpu=True\nbf16=True\nfp16=True\n')
    st.success("AMD setup complete! 🚀")

def setup_cpu():
    st.subheader("⬜ CPU-Only Detected")
    st.info("No supported GPU detected. Running on CPU.")
    run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], show_output=True)
    # Set flags for CPU: bf16 if available, fp16 off
    with open('FLAGS.txt', 'w') as f:
        f.write('use_gpu=False\nbf16=True\nfp16=False\n')
    st.success("CPU setup complete! 🚀")
    os.system("streamlit run rhapsody.py")

# ---------- GPU Detection ----------

def get_gpu_info():
    try:
        # NVIDIA
        nvidia = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if nvidia.returncode == 0 and nvidia.stdout.strip():
            setup_nvidia()
            return f"NVIDIA GPU detected: {nvidia.stdout.strip()}"
        # AMD
        lspci = subprocess.run(['lspci'], stdout=subprocess.PIPE, text=True)
        amd_gpus = [line for line in lspci.stdout.split('\n') if 'AMD' in line or 'Advanced Micro Devices' in line]
        if amd_gpus:
            setup_amd()
            return "AMD GPU(s) detected:\n" + "\n".join(amd_gpus)
        # Other GPU
        gpus = [line for line in lspci.stdout.split('\n') if 'VGA' in line or '3D' in line]
        if gpus:
            setup_cpu()
            return "Other GPU(s) detected:\n" + "\n".join(gpus)
        setup_cpu()
        return "No GPU detected on this system."
    except Exception as e:
        setup_cpu()
        return f"Error detecting GPU: {e}"

# ---------- Main App ----------

def main():
    st.markdown("""
        <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {font-size: 1.2em; padding: 0.5em 2em;}
        .stAlert {font-size: 1.1em;}
        </style>
        """, unsafe_allow_html=True)
    st.title("🚀 GPU Detection & Auto-Setup")
    st.markdown("""
    <div style='font-size:1.2em;'>
    <b>Welcome to Rhapsody, where local AI meets small-scale innovation!</b> This tool will:<br>
    • Detect your GPU and OS, and install deep learning frameworks like PyTorch, and configure them based on your GPU automatically<br>
    • Use HF Transformers for a wide range of models, but use Ollama/LlamaCPP as a fallback<br>
    • Allow you to generate images using diffusion models of your choice (not all might be compatible)
    <br>
    <span style='color:#e67e22;'>⚠️ You may be prompted for your password!</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown(f"**Detected OS:** `{detect_os()}`")
    if check_amd_windows():
        st.stop()
    if st.button("🔍 Detect and Auto-Install GPU/CPU Support", use_container_width=True):
        with st.spinner("Detecting GPU and setting up your environment..."):
            time.sleep(1)
            gpu_info = get_gpu_info()
            st.info(gpu_info)
    if st.button("Donate to the dev"):
        st.write("Thank you so much for donating!")
        st.write("**Bitcoin**: 1E5SVrJfaAvVdRTSXkHsgx2BG8SebXfatS")
        st.write("**Monero**: 46KeFZzY8arRhU9w67DpKpDMjns6NwRkMS6gtrZ5vrjDJ3EfSDs8CxLGbQEG87G5bUW7SVxVykgUEEsnfAxTXVdx2WGVefY")
        st.write("**Litecoin**: LhPMt8XQXSMRg6m4zHa6PssTkLc5B72mi4")
        st.write("**Ethereum**: 0xD5Bd30408493Ce7255eB3A01D5e0f907e22F9727")
        st.write("**XRP**: rLpW3b6T3KriK26bU2d5u2gGQZ2MpavBDqr")
if __name__ == "__main__":
    main()