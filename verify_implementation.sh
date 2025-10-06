#!/bin/bash

echo "=========================================="
echo "ML-AI Framework Implementation Verification"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (missing)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${RED}✗${NC} $1/ (missing)"
        return 1
    fi
}

echo "1. Core Models"
check_file "src/models/__init__.py"
check_file "src/models/schemas.py"
echo ""

echo "2. Custom Tools"
check_file "src/tools/__init__.py"
check_file "src/tools/data_tools.py"
check_file "src/tools/ml_tools.py"
check_file "src/tools/analysis_tools.py"
echo ""

echo "3. Workflows"
check_file "src/workflows/__init__.py"
check_file "src/workflows/crew_system.py"
check_file "src/workflows/langgraph_system.py"
echo ""

echo "4. Utilities"
check_file "src/utils/__init__.py"
check_file "src/utils/logging.py"
check_file "src/utils/error_handling.py"
echo ""

echo "5. AG-UI Server"
check_file "src/ag_ui_server.py"
echo ""

echo "6. Configuration"
check_file "pyproject.toml"
check_file "requirements.txt"
check_file ".env.example"
check_file "config/settings.py"
echo ""

echo "7. Development Tools"
check_file "Makefile"
check_file ".gitignore"
check_file "README.md"
echo ""

echo "8. Examples"
check_file "examples/simple_workflow.py"
check_file "examples/langgraph_workflow.py"
echo ""

echo "9. Documentation"
check_file "docs/IMPLEMENTATION_SUMMARY.md"
echo ""

echo "10. Code Statistics"
echo "-------------------"
TOTAL_LINES=$(find src -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
TOTAL_FILES=$(find src -name "*.py" | wc -l)
echo "Total Python files in src/: $TOTAL_FILES"
echo "Total lines of code: $TOTAL_LINES"
echo ""

echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
