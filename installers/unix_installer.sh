#!/bin/bash
set -e

#* Terminal Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}--- EmoProsopon Installer ---${NC}"

REPO_URL="https://github.com/CrawlingWharf90/EmoProsopon.git"
INSTALL_DIR="$HOME/.emoprosopon"

#? 1. Check for Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: 'git' is not installed on this system.${NC}"
    echo -e "Please install Git to continue the installation:"
    echo -e "  - ${CYAN}Ubuntu/Debian:${NC} sudo apt install git"
    echo -e "  - ${CYAN}macOS:${NC} brew install git ${YELLOW}(or run: xcode-select --install)${NC}"
    echo -e "  - ${CYAN}Official Download:${NC} https://git-scm.com/downloads"
    echo ""
    exit 1
fi

#? 2. Check for Python 3
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed on this system.${NC}"
    echo -e "EmoProsopon requires Python 3.12+ to run. Please install it:"
    echo -e "  - ${CYAN}Arch Linux:${NC} sudo pacman -S python"
    echo -e "  - ${CYAN}Ubuntu/Debian:${NC} sudo apt install python3"
    echo -e "  - ${CYAN}macOS:${NC} brew install python3"
    echo -e "  - ${CYAN}Official Download:${NC} https://www.python.org/downloads/"
    echo ""
    exit 1
fi

#? 3. Prompt
echo -e "This will download EmoProsopon to ${YELLOW}$INSTALL_DIR${NC}"
read -p "Do you want to proceed? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Installation aborted.${NC}"
    exit 1
fi

echo -e "\n${CYAN}Downloading from GitHub...${NC}"

#? 4. Clone or Update
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing old installation..."
    rm -rf "$INSTALL_DIR"
fi

#? Use --depth 1 for a lightning-fast download that ignores git history
git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"

#? 5. Permissions and Symlink
echo -e "\n${CYAN}Setting up global command...${NC}"
chmod +x "$INSTALL_DIR/eop.py"

echo "Creating global 'eop' command (may require password for sudo)..."
sudo ln -sf "$INSTALL_DIR/eop.py" /usr/local/bin/eop

echo -e "\n${GREEN}Installation Complete!${NC}"
echo -e "Run ${CYAN}eop --setup${NC} to initialize your environment."