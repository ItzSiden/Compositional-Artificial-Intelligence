#!/bin/bash
# MSCP Setup Script for GitHub Codespaces
set -e

echo "=================================================="
echo " MSCP - Multi-System Compositional Pipeline Setup"
echo "=================================================="

# Step 1: Create virtual environment
echo ""
echo "[Phase 1] Creating Python virtual environment..."
python3 -m venv mscp_env
source mscp_env/bin/activate

# Step 2: Install all dependencies
echo ""
echo "[Phase 1] Installing core dependencies..."
pip install --upgrade pip -q

echo "  → Installing llama-cpp-python (CPU only)..."
pip install llama-cpp-python -q

echo "  → Installing FAISS + sentence-transformers..."
pip install faiss-cpu sentence-transformers -q

echo "  → Installing NetworkX + matplotlib..."
pip install networkx matplotlib -q

echo "  → Installing other utilities..."
pip install huggingface_hub tqdm colorama -q

echo ""
echo "[Phase 3] Downloading AI model (Llama-3.2-1B)..."
echo "  This may take a few minutes (~700MB)..."

python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import os

model_path = "models"
os.makedirs(model_path, exist_ok=True)

# Download Llama 3.2 1B quantized model
try:
    path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        local_dir=model_path
    )
    print(f"  ✓ Model downloaded to: {path}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    print("  → Please manually download from: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
    print("  → Place the .gguf file in the 'models/' folder")
EOF

echo ""
echo "=================================================="
echo " Setup Complete!"
echo ""
echo " To activate and run:"
echo "   source mscp_env/bin/activate"
echo "   python main.py"
echo "=================================================="
