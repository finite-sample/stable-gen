#!/bin/bash

# Deterministic Training - Automated Runner
# This script helps you run experiments easily

set -e

echo "=================================="
echo "Deterministic Training Experiments"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

# Function to run quick start
run_quick_start() {
    echo -e "${GREEN}Running quick start demo...${NC}"
    echo "This takes ~5 minutes on GPU, ~15 minutes on CPU"
    echo ""
    python3 quick_start.py
}

# Function to run full demo
run_full_demo() {
    echo -e "${GREEN}Running full experiment...${NC}"
    echo "This takes ~15 minutes on GPU, ~30+ minutes on CPU"
    echo ""
    python3 deterministic_training_demo.py
}

# Function to check dependencies
check_deps() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    python3 -c "import torch" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}PyTorch not installed${NC}"
        echo "Install with: pip install torch"
        return 1
    fi
    
    python3 -c "import transformers" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}Transformers not installed${NC}"
        echo "Install with: pip install transformers"
        return 1
    fi
    
    echo -e "${GREEN}✓ All dependencies installed${NC}"
    return 0
}

# Function to install dependencies
install_deps() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Installation complete${NC}"
}

# Function to show menu
show_menu() {
    echo ""
    echo "Select an option:"
    echo "  1) Quick start (5-10 min demo)"
    echo "  2) Full experiment (15-30 min with plots)"
    echo "  3) Check dependencies"
    echo "  4) Install dependencies"
    echo "  5) Open Jupyter notebook"
    echo "  6) View project overview"
    echo "  7) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-7]: " choice
    
    case $choice in
        1)
            if check_deps; then
                run_quick_start
            else
                echo -e "${YELLOW}Install dependencies first (option 4)${NC}"
            fi
            ;;
        2)
            if check_deps; then
                run_full_demo
            else
                echo -e "${YELLOW}Install dependencies first (option 4)${NC}"
            fi
            ;;
        3)
            check_deps
            ;;
        4)
            install_deps
            ;;
        5)
            echo -e "${GREEN}Starting Jupyter notebook...${NC}"
            jupyter notebook deterministic_training.ipynb
            ;;
        6)
            if command -v less &> /dev/null; then
                less PROJECT_OVERVIEW.md
            else
                cat PROJECT_OVERVIEW.md
            fi
            ;;
        7)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
