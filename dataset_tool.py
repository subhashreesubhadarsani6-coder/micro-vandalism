#!/usr/bin/env python
"""
Guarding the Stones — Dataset Preparation & Annotation CLI Runner.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
