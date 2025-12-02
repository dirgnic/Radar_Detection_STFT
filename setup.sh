#!/bin/bash
# Setup script pentru proiectul Sistem Radar
# Creează virtual environment și instalează dependențe

echo "=================================================="
echo "  SETUP SISTEM RADAR - DETECTARE AERONAVE"
echo "=================================================="
echo ""

# Verificare Python
echo "[1/6] Verificare Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 nu este instalat!"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "✓ $PYTHON_VERSION detectat"
echo ""

# Creare virtual environment
echo "[2/6] Creare virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment există deja. Șterge? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf venv
        echo "✓ Virtual environment vechi șters"
    else
        echo "✓ Se folosește virtual environment existent"
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment creat: venv/"
fi
echo ""

# Activare virtual environment
echo "[3/6] Activare virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activat"
echo ""

# Upgrade pip
echo "[4/6] Upgrade pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip actualizat"
echo ""

# Instalare dependențe Python
echo "[5/6] Instalare dependențe Python..."
echo "    - numpy (calcul numeric)"
echo "    - scipy (procesare semnal)"
echo "    - matplotlib (vizualizări)"
echo "    - seaborn (grafice avansate)"
echo "    - pandas (analiză date)"

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Toate dependențele Python au fost instalate"
else
    echo "❌ Eroare la instalarea dependențelor"
    exit 1
fi
echo ""

# Verificare Haskell (opțional)
echo "[6/6] Verificare Haskell (opțional pentru optimizări)..."
if command -v ghc &> /dev/null; then
    GHC_VERSION=$(ghc --version)
    echo "✓ $GHC_VERSION detectat"
    
    # Compilare module Haskell
    if [ -d "haskell_optimize" ]; then
        echo "  Compilare module Haskell de optimizare..."
        cd haskell_optimize
        if [ -f "compile.sh" ]; then
            bash compile.sh
            if [ $? -eq 0 ]; then
                echo "  ✓ Module Haskell compilate"
            fi
        fi
        cd ..
    fi
else
    echo "⚠️  Haskell nu este instalat (opțional)"
    echo "    Pentru instalare: https://www.haskell.org/downloads/"
fi
echo ""

# Creare directoare necesare
echo "Creare structură directoare..."
mkdir -p results
mkdir -p latex_docs/output
mkdir -p research/papers
echo "✓ Directoare create"
echo ""

# Testare instalare
echo "=================================================="
echo "  TESTARE INSTALARE"
echo "=================================================="
echo ""
python tests/test_radar.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✓ SETUP COMPLET - SUCCES!"
    echo "=================================================="
    echo ""
    echo "Pentru a activa virtual environment în viitor:"
    echo "  source venv/bin/activate"
    echo ""
    echo "Pentru a rula aplicația:"
    echo "  python main.py"
    echo ""
    echo "Pentru simulări:"
    echo "  python simulations/single_target.py"
    echo "  python simulations/multiple_targets.py"
    echo "  python simulations/moving_targets.py"
    echo ""
    echo "Pentru documentație LaTeX:"
    echo "  cd latex_docs && bash compile_latex.sh"
    echo ""
else
    echo ""
    echo "⚠️  Testele au eșuat. Verificați instalarea."
fi
