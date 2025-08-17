#!/bin/bash
# Activate virtual environment
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo ""
echo "To deactivate, run: deactivate"
echo "To install new packages: pip install package_name"
echo "To save dependencies: pip freeze > requirements.txt"
