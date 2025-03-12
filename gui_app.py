#!/usr/bin/env python3
"""
GUI application entry point for the Check Extractor.
"""

import os
import sys
from src.gui import main

if __name__ == "__main__":
    # Ensure we're running from the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Run the GUI application
    main()
