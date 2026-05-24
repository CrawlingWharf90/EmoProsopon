#!/bin/bash
set -e

# Terminal Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}--- EmoProsopon Installer ---${NC}"

# 1. Dynamically calculate the size of the repository
# 'du -sh' gets the human-readable size (e.g., 45M, 1.2G) of the parent folder
REPO_SIZE=$(du -sh ../ | awk '{print $1}')

echo -e "Estimated Installation Space Required: ${YELLOW}${REPO_SIZE}${NC}\n"

# 2. Prompt the user for confirmation
read -p "Do you want to proceed with the installation? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Installation aborted by user.${NC}"
    exit 1
fi

echo -e "\n${CYAN}Installing EmoProsopon...${NC}"

INSTALL_DIR="$HOME/.emoprosopon"

# 3. Copy files to the local directory
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation..."
    rm -rf "$INSTALL_DIR"
fi

echo "Copying files to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
cp -r ../* "$INSTALL_DIR/"

# 4. Make the orchestrator executable
chmod +x "$INSTALL_DIR/eop.py"

# 5. Create the global symlink
echo "Creating global 'eop' command (may require password for sudo)..."
sudo ln -sf "$INSTALL_DIR/eop.py" /usr/local/bin/eop

echo -e "\n${GREEN}Installation Complete!${NC}"
echo -e "Run ${CYAN}eop --setup${NC} to initialize your environment."