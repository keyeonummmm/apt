#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}Starting APT Viewer...${NC}"

# Check if virtual environment exists
if [ ! -d "${SCRIPT_DIR}/my_env" ]; then
    echo -e "${RED}Error: Virtual environment not found at ${SCRIPT_DIR}/my_env${NC}"
    echo "Please make sure you have set up the virtual environment correctly."
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "${SCRIPT_DIR}/my_env/bin/activate"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    exit 1
fi

# Navigate to the finetuning directory
cd "${SCRIPT_DIR}/finetuning"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to navigate to finetuning directory${NC}"
    deactivate
    exit 1
fi

# Run the viewer
echo -e "${GREEN}Launching APT Viewer...${NC}"
python apt_viewer.py

# Capture exit code
EXIT_CODE=$?

# Deactivate virtual environment
deactivate

# Check if the program exited normally
if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Error: APT Viewer exited with code ${EXIT_CODE}${NC}"
    exit $EXIT_CODE
fi

exit 0 