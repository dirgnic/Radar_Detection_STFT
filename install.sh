#!/bin/bash

# Setup script for Radar Detection System
# Author: Ingrid Corobana
# Date: December 2025

echo "=================================================="
echo "  RADAR DETECTION SYSTEM - SETUP"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}✗ Python 3.8+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}→ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create results directory
echo ""
echo "Creating directories..."
mkdir -p results
mkdir -p results/figures
mkdir -p results/data
echo -e "${GREEN}✓ Directories created${NC}"

# Run tests
echo ""
echo "Running tests..."
if python tests/test_radar.py > /dev/null 2>&1; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed (check manually)${NC}"
fi

# Check for Haskell (optional)
echo ""
echo "Checking for Haskell (optional)..."
if command -v ghc &> /dev/null; then
    GHC_VERSION=$(ghc --version | awk '{print $NF}')
    echo -e "${GREEN}✓ GHC $GHC_VERSION found${NC}"
    echo "  → Haskell optimizations available"
else
    echo -e "${YELLOW}→ GHC not found (Haskell optimizations disabled)${NC}"
    echo "  → Install with: brew install ghc cabal-install (macOS)"
fi

# Check for LaTeX (optional)
echo ""
echo "Checking for LaTeX (optional)..."
if command -v pdflatex &> /dev/null; then
    echo -e "${GREEN}✓ LaTeX found${NC}"
    echo "  → Can compile presentation and paper"
else
    echo -e "${YELLOW}→ LaTeX not found (document compilation disabled)${NC}"
    echo "  → Install with: brew install --cask mactex (macOS)"
fi

# Summary
echo ""
echo "=================================================="
echo "  SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:  source venv/bin/activate"
echo "  2. Run main application:  python main.py"
echo "  3. Run single simulation: python simulations/single_target.py"
echo "  4. Run all tests:         python tests/test_radar.py"
echo ""
echo "Documentation:"
echo "  - Quick start:            QUICKSTART.md"
echo "  - Full documentation:     DOCUMENTATION.md"
echo "  - Research papers:        research/"
echo ""
echo "=================================================="
